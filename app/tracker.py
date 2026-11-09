import math

class CentroidTracker:
    """Small local tracker; IDs are approximate and may switch after occlusion."""
    def __init__(self,max_distance=100,max_disappeared=25):
        self.next_id=1; self.objects={}; self.missing={}
        self.max_distance=max_distance; self.max_disappeared=max_disappeared

    def update(self, boxes):
        centers=[((b[0]+b[2])//2,(b[1]+b[3])//2) for b in boxes]
        result={}; used=set()
        for i,c in enumerate(centers):
            choices=[(math.dist(c,old),oid) for oid,old in self.objects.items() if oid not in used]
            choices.sort()
            if choices and choices[0][0] <= self.max_distance:
                oid=choices[0][1]; used.add(oid)
            else:
                oid=self.next_id; self.next_id+=1
            self.objects[oid]=c; self.missing[oid]=0; result[i]=oid
        for oid in list(self.objects):
            if oid not in used and oid not in result.values():
                self.missing[oid]=self.missing.get(oid,0)+1
                if self.missing[oid]>self.max_disappeared:
                    self.objects.pop(oid,None); self.missing.pop(oid,None)
        return result
