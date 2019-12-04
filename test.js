var Lightcube = require("./Lightcube.js");
var tf = require("@tensorflow/tfjs-node");

var cubeSize = 32;
var kernelSize = 1; //half size

var lc = new Lightcube(cubeSize, kernelSize);

for(var i = 0; i < 100; i++)
{
  console.log(i);
  tf.tidy(() =>
  {
    lc.step();
    lc.train(1);
  })
}
