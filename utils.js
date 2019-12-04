function shift(t, n)
{
  //TODO: how to wrap
  var origShape = t.shape;
  var n2 = n.map((e) => [Math.abs(e), Math.abs(e)]);
  n = n.map((e) => e < 0 ? 0 : e * 2);
  t = t.pad(n2);
  t = t.slice(n, origShape);
  return t;
}

function for3d(x1, y1, z1, x2, y2, z2, f)
{
  for(var x = x1; x < x2; x++)
  {
    for(var y = y1; y < y2; y++)
    {
      for(var z = z1; z < z2; z++)
      {
        f(x, y, z);
      }
    }
  }
}

module.exports = {shift, for3d};
