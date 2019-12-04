var Lightcube = require("./Lightcube.js");
var ProtoPost = require("protopost");
var express = require("express");
var tf = require("@tensorflow/tfjs-node");

var app = express();
app.use(express.static('public'));

var cubeSize = 16;
var kernelSize = 1; //half size

// var lc = new Lightcube({
//   cubeSize: cubeSize,
//   kernelSize: kernelSize,
//   potentialLeakRate: 0.1,
//   sensitivityIncrease: 0.0001,
//   sensitivityDecrease: 0.1,
//   spikeDecayRate: 0.01,
//   traceDecayRate: 0.1,
//   //learningRate: 1e-4,
//   learningRate: 1e-1,
//   synMin: 0,
//   synMax: 1,
//   traceDir: 1,
//   potentialIncrease: 0,//0.000001
// });

var lc = new Lightcube({
  cubeSize: cubeSize,
  kernelSize: kernelSize,
  potentialLeakRate: 0.001,
  sensitivityIncrease: 0.0001,
  sensitivityDecrease: 0.1,
  spikeDecayRate: 0.01,
  traceDecayRate: 0.1,
  //learningRate: 1e-4,
  learningRate: 1e-1,
  synMin: 0,
  synMax: 1,
  traceDir: 1,
  //potentialIncrease: 0.000001
  potentialIncrease: 0.01
});

var reward = 0;

function update()
{
  tf.tidy(() =>
  {
    //lc.train(reward);
    lc.train(1);
    var s = Math.floor(cubeSize / 2);
    var one = tf.tensor([[[1]]]).pad([[s-1, s], [s-1, s], [s-1, s]])
    //var one = tf.tensor([[[1]]]).pad([[31, 32], [31, 32], [31, 32]])
    lc.potential.assign(lc.potential.add(one));
    lc.step();
    reward = 0;

    console.log(lc.potential.mean().dataSync(), tf.memory().numTensors)
  });

  setImmediate(update);
}

update();

var api = new ProtoPost({
  getSize: (data) => cubeSize,
  getSpikes: (data) => lc.spikes.arraySync(),
  getSpikesFlat: (data) => Array.from(lc.spikes.dataSync()),
  //getSpikesFlat: (data) => Array.from(tf.tidy(() => tf.ones([cubeSize, cubeSize, cubeSize]).dataSync())),
  getPotential: (data) => lc.potential.arraySync(),
  reward: (data) => reward += data.reward,
});

app.use("/api", api.router);

app.listen(3000, () => console.log("Listening on port 3000!"))
