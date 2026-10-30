"""Same tracker dynamics, fused guarded GPU peak decoder and actual peak evidence."""
import copy,time
import numpy as np
import torch
from .finder import BallFinder,decode_peaks,event_grid
from .finder_fused_v230 import FusedFinderGraph
from .multiscale_event_tracker_v4 import LatestEventTracker


class FusedEventTracker(LatestEventTracker):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.cuda.set_per_process_memory_fraction(1*2**30/torch.cuda.get_device_properties(0).total_memory)
        self.finder=BallFinder().eval()
        self.finder.load_state_dict(torch.load('outputs/models/finder_v2.pt',map_location='cpu',weights_only=True))
        self.finder.cuda();self.numeric=[];self.numeric_peaks=[];self.forward_ms=[];self.fallback_history=[];self.peak_evidence=None
        start=time.perf_counter()
        with torch.inference_mode():
            sample=torch.zeros((2,6,120,160),device='cuda')
            for _ in range(5):tuple(x.cpu() for x in self.finder(sample))
        self.warmup_seconds=time.perf_counter()-start
        self.graph_runner=FusedFinderGraph(self.finder,self.max_balls,self.peak_min)
        with torch.inference_mode():
            for _ in range(3):
                raw,packet=self.graph_runner(sample)
                self.graph_runner.unpack(raw,packet,True)
        self.last_segments={}
        self.warmup_seconds=time.perf_counter()-start

    @torch.inference_mode()
    def step(self,sim,t):
        self.processed=False
        if not self.on:
            if t>=self.t_rec-1e-9:self.rig.reset(t);self.on=True
            return
        evs=self.rig.step(t)
        ingest_wall=time.perf_counter();ingest_cpu=time.thread_time()
        for c,e in enumerate(evs):
            self.hist[c].append((e['x'],e['y'],np.round(e['t']*1e6).astype(np.int32),e['p']))
        self.last_segments=dict(ingest_wall_ms=1000*(time.perf_counter()-ingest_wall),ingest_thread_ms=1000*(time.thread_time()-ingest_cpu),processed=False)
        if t<0 or t+1e-9<self.next_process or not self.processing_allowed:return
        grid_wall=time.perf_counter();grid_cpu=time.thread_time()
        grids=[]
        for history in self.hist:
            x,y,ts,p=(np.concatenate(a) for a in zip(*history))
            grids.append(event_grid(x,y,ts,p,round(t*1e6)))
        grid=np.stack(grids)
        self.last_segments.update(grid_wall_ms=1000*(time.perf_counter()-grid_wall),grid_thread_ms=1000*(time.thread_time()-grid_cpu))
        begin_event=torch.cuda.Event(enable_timing=True);end_event=torch.cuda.Event(enable_timing=True)
        start=time.perf_counter();forward_cpu=time.thread_time();begin_event.record()
        raw,packet=self.graph_runner(torch.from_numpy(grid).cuda());end_event.record()
        save_raw=len(self.numeric)<32
        peaks,audit_raw,fallback=self.graph_runner.unpack(raw,packet,save_raw)
        self.forward_ms.append(1000*(time.perf_counter()-start))
        self.last_segments.update(forward_wall_ms=self.forward_ms[-1],forward_thread_ms=1000*(time.thread_time()-forward_cpu),forward_device_elapsed_ms=begin_event.elapsed_time(end_event),fused_peak_decode=True,CPU_peak_fallback=fallback,raw_audit_transfer=save_raw)
        if save_raw:
            self.numeric.append((grid.copy(),audit_raw[0],audit_raw[1]))
            self.numeric_peaks.append(([v.copy() for v in peaks],fallback))
        self.fallback_history.append(dict(t=t,cameras=fallback))
        # GPU decode and optional original CPU fallback are included above, never charged twice.
        self.last_segments.update(decode_wall_ms=0.,decode_thread_ms=0.)
        track_wall=time.perf_counter();track_cpu=time.thread_time()
        for tr in self.tracks:tr.ekf.predict(t)
        dets=self._associate(peaks,t);self._spawn(peaks,t)
        self.tracks=[tr for tr in self.tracks if t-tr.last_seen<=self.max_miss]
        for tr in self.tracks:
            if tr.id in dets:tr.ekf.update(dets[tr.id],self.score_min)
        if self.first_det_t is None and self.tracks:self.first_det_t=t
        self._predict(dets,t)
        self.last_segments.update(tracking_forecast_wall_ms=1000*(time.perf_counter()-track_wall),tracking_forecast_thread_ms=1000*(time.thread_time()-track_cpu),processed=True)
        self.next_process=t+self.forecast_period;self.processed=True
        self.log.append(dict(t=t,peaks=peaks,events=evs,bound=None,
            tracks={tr.id:tr.ekf.x[:3].copy() for tr in self.tracks},
            candidates={tr.id:dict(hits=tr.hits,speed=float(np.linalg.norm(tr.ekf.x[3:6])),
                path=tr.outs[-1]['path']) for tr in self.tracks if tr.outs}))

    def verify_raw(self):
        # Post-trial audit has no access to physical labels and never controls the robot.
        reference=copy.deepcopy(self.finder).cpu().eval();maximum=0.;pixel_delta=0.;same_counts=True
        with torch.inference_mode():
            for grid,heat,offset in self.numeric:
                out=reference(torch.from_numpy(grid));ref=[x.numpy() for x in out]
                maximum=max(maximum,float(np.abs(ref[0]-heat).max()),float(np.abs(ref[1]-offset).max()))
                a=decode_peaks(*out,self.max_balls,self.peak_min)
                b=decode_peaks(torch.from_numpy(heat),torch.from_numpy(offset),self.max_balls,self.peak_min)
                for u,v in zip(a,b):
                    if len(u)!=len(v):same_counts=False
                    elif len(u):pixel_delta=max(pixel_delta,float(np.abs(u[:,:2]-v[:,:2]).max()))
        executed_xy=executed_score=0.;executed_counts=True
        for (_,heat,offset),(actual,flags) in zip(self.numeric,self.numeric_peaks):
            expected=decode_peaks(torch.from_numpy(heat),torch.from_numpy(offset),self.max_balls,self.peak_min)
            for a,b in zip(expected,actual):
                if a.shape!=b.shape:executed_counts=False
                elif len(a):
                    executed_xy=max(executed_xy,float(np.abs(a[:,:2]-b[:,:2]).max()))
                    executed_score=max(executed_score,float(np.abs(a[:,2]-b[:,2]).max()))
        return dict(frontend='fused_guarded_GPU_peaks_v230',accepted=bool(len(self.numeric)>=16 and maximum<=5e-5 and same_counts and pixel_delta<=.01 and len(self.numeric_peaks)==len(self.numeric) and executed_counts and executed_xy<=.01 and executed_score<=5e-6),
            raw_batches=len(self.numeric),max_absolute_output_delta=maximum,decoded_peak_counts_equal=same_counts,
            max_decoded_peak_pixel_delta=pixel_delta,physical_gpu=0,memory_cap_gib=1,
            actual_executed_peak_batches=len(self.numeric_peaks),actual_executed_peak_counts_equal=executed_counts,
            actual_executed_peak_max_pixel_delta=executed_xy,actual_executed_peak_max_score_delta=executed_score,
            executed_peak_evidence=self.peak_evidence,CPU_fallback_cameras=sum(sum(x['cameras'])for x in self.fallback_history),
            forward_plus_transfer_ms=dict(median=float(np.median(self.forward_ms)),p95=float(np.percentile(self.forward_ms,95))),
            warmup_seconds=self.warmup_seconds,scope='Post-trial same-weight CPU/GPU raw and actually executed fused/fallback peak check; no physical labels or controller actions')
