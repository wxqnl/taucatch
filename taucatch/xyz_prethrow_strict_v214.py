"""Never deliver even a sub-nanosecond-future estimate to strict context."""
import heapq

from .multiscale_v1 import DeliveryQueue
from .xyz_prethrow_observation_v202 import SlottedObservation


class StrictDeliveryQueue(DeliveryQueue):
    def deliver(self,t):
        while self.heap and self.heap[0][0] <= t:
            _,_,forecast=heapq.heappop(self.heap)
            self.latest=forecast
        return self.latest


class StrictSlottedObservation(SlottedObservation):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.queue=StrictDeliveryQueue()
