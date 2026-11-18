"""Host-only model of V2 producer/consumer coverage; not a device test."""
import unittest
from collections import Counter

CORES, WIDTH, BM, BN, READY = 8, 2048, 128, 256, 512

def tiles(rows):
    for s in range((rows + READY - 1) // READY):
        for c in range(CORES):
            m = s * READY + (c // 2) * BM
            if m >= rows:
                continue
            for j in range(4):
                w = 3 - j if s & 1 else j
                yield s, c, m, (w * 2 + c % 2) * BN

class TestV2(unittest.TestCase):
    def test_tiles(self):
        for rows in range(1024, 4097, 256):
            seen = Counter((m, n) for _, _, m, n in tiles(rows))
            self.assertEqual(seen, Counter({(m,n): 1 for m in range(0,rows,BM)
                                          for n in range(0,WIDTH,BN)}))
            for s,c,m,n in tiles(rows):
                self.assertEqual(m//READY,s)
                self.assertLessEqual((m+BM)*WIDTH*2, rows*WIDTH*2)
                self.assertLess(n*16, WIDTH*WIDTH)
                self.assertLessEqual((m+BM-1)*WIDTH+n+BN,rows*WIDTH)
            # flags: eight cache-line seats per slab, within 8 KiB suffix
            self.assertLessEqual(((rows+READY-1)//READY)*CORES*64, 8192)

    def test_consumers(self):
        for rows in range(1024, 4097, 256):
            for sub in (512,1024):
                seen = []
                for lo in range(0,rows,sub):
                    real = min(sub,rows-lo)
                    ns = (real+READY-1)//READY
                    seen.extend(range(lo//READY,lo//READY+ns))
                    self.assertEqual((real*WIDTH*2//4)%16384,0)
                    self.assertEqual((real*WIDTH*2//2)%512,0)
                self.assertEqual(seen, list(range((rows+READY-1)//READY)))

if __name__ == "__main__":
    unittest.main()
