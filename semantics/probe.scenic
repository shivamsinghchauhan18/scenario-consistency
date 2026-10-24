# Probe Scenic's positional and heading semantics in the frame ChatScene uses:
# ego at the origin facing "forward" (+y), a reference point 20 m ahead.
ego = new Object at (0, 0), facing 0 deg, with allowCollisions True
P = new OrientedPoint at (0, 20), facing 0 deg
leftNeg = new Object left of P by -5, with width 0.5, with length 0.5, with allowCollisions True     # ChatScene spawn 1/7/9/11 pattern
leftPos = new Object left of P by 5, with width 0.5, with length 0.5, with allowCollisions True
rightPos = new Object right of P by 5, with width 0.5, with length 0.5, with allowCollisions True    # spawn 2/6/8/10 pattern
plus90 = new Object at (30, 0), facing P.heading + 90 deg, with width 0.5, with length 0.5, with allowCollisions True
minus90 = new Object at (-30, 0), facing P.heading - 90 deg, with width 0.5, with length 0.5, with allowCollisions True
