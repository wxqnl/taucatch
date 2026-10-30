"""Opt-in candidate crop magnification for controlled chooser experiments.

Keeps mosaic slot order and redraws the same numbered marks. Existing training, serving and
rendering do not call this helper; a new model needs matching train/serve preprocessing.
"""
import numpy as np
from PIL import Image, ImageDraw

from .choice import MAX_CANDIDATES, VIEW, _font


def mosaic_candidate_count(mosaic):
    """Numbered occupied cells remain observable even when a track's speed is zero."""
    cell=VIEW//2
    occupied=[bool(np.any(mosaic[(k//2)*cell:(k//2+1)*cell,(k%2)*cell:(k%2+1)*cell]))
              for k in range(MAX_CANDIDATES)]
    n=sum(occupied)
    if occupied != [k<n for k in range(MAX_CANDIDATES)]:
        raise ValueError('Noncontiguous mosaic slots')
    return n


def zoom_mosaic(mosaic, n_candidates, factor=2):
    mosaic=np.asarray(mosaic)
    if mosaic.shape!=(VIEW,VIEW,3) or mosaic.dtype!=np.uint8:
        raise ValueError('Expected uint8 224x224 RGB mosaic')
    if not 0<=n_candidates<=MAX_CANDIDATES or factor not in (1,2,4):
        raise ValueError('Invalid candidate count or zoom factor')
    if factor==1:return mosaic.copy()
    cell=VIEW//2;extent=cell//factor;offset=(cell-extent)//2
    result=Image.new('RGB',(VIEW,VIEW),(0,0,0))
    for k in range(n_candidates):
        x,y=(k%2)*cell,(k//2)*cell
        crop=Image.fromarray(mosaic[y+offset:y+offset+extent,x+offset:x+offset+extent])
        crop=crop.resize((cell,cell),Image.Resampling.BILINEAR)
        ImageDraw.Draw(crop).text((4,2),str(k+1),fill=(255,255,255),font=_font(16),
                                 stroke_width=2,stroke_fill=(0,0,0))
        result.paste(crop,(x,y))
    return np.array(result, copy=True)
