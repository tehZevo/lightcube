var tf = require("@tensorflow/tfjs-node");
var U = require("./utils.js");

module.exports = class Lightcube
{
  constructor(o)
  {
    //parameters
    this.cubeSize = o.cubeSize
    this.kernelSize = o.kernelSize
    this.cubeShape = [o.cubeSize, o.cubeSize, o.cubeSize];
    this.potentialLeakRate = o.potentialLeakRate
    this.sensitivityIncrease = o.sensitivityIncrease
    this.sensitivityDecrease = o.sensitivityDecrease
    this.spikeDecayRate = o.spikeDecayRate
    this.traceDecayRate = o.traceDecayRate
    this.learningRate = o.learningRate
    this.synMin = o.synMin
    this.synMax = o.synMax
    this.traceDir = o.traceDir
    this.potentialIncrease = o.potentialIncrease;

    this.reset()
  }

  reset()
  {
    this.potential = tf.variable(tf.zeros(this.cubeShape));
    //this.sensitivity = tf.variable(tf.zeros(this.cubeShape));
    this.sensitivity = tf.variable(tf.ones(this.cubeShape));
    this.spikes = tf.variable(tf.zeros(this.cubeShape, "bool"));
    this.spikeDecay = tf.variable(tf.zeros(this.cubeShape));
    var synapses = {};
    var traces = {};
    var ks = this.kernelSize;
    //TODO: convert to a single tensor with reshapes n stuff
    U.for3d(-ks, -ks, -ks, ks+1, ks+1, ks+1, (x, y, z) =>
    {
      //var w = tf.variable(tf.zeros(this.cubeShape));
      var w = tf.variable(tf.randomUniform(this.cubeShape, 0, 1));
      synapses[[x, y, z]] = w;
      var t = tf.variable(tf.zeros(this.cubeShape));
      traces[[x, y, z]] = t;
    });
    this.synapses = synapses;
    this.traces = traces;
  }

  step()
  {
    var ks = this.kernelSize;
    //probabilistic spiking threshold
    var threshold = tf.randomUniform(this.cubeShape, 0, 1);
    //update spikes
    this.spikes.assign(tf.greater(this.potential, threshold));
    var potential = this.potential;
    //reset potential where spikes occur
    potential = tf.where(this.spikes, tf.zerosLike(potential), potential);
    //decay potential
    potential = potential.mul(1 - this.potentialLeakRate);
    //this.potential += tf.tensordot(tf.cast(this.spikes, "float32"), this.synapses, 1) / this.size * this.sensitivity

    var spikeCounter = tf.zeros(this.cubeShape);
    U.for3d(-ks, -ks, -ks, ks+1, ks+1, ks+1, (x, y, z) =>
    {
      //shift spikes to be local to neuron
      var spikeShift = U.shift(this.spikes, [x, y, z]);
      //count spikes for sensitivity calculation later
      spikeCounter = spikeCounter.add(spikeShift);
      //add to potential
      var spikeSynSens = this.spikes.mul(this.synapses[[x, y, z]]).mul(this.sensitivity);
      potential = this.potential.add(spikeSynSens);
    });

    //increase potential slowly over time
    potential = this.potential.add(this.sensitivity.mul(this.potentialIncrease));
    this.potential.assign(potential);

    //update sensitivity
    var sens = this.sensitivity
    var oneSubDec = tf.onesLike(spikeCounter).sub(this.sensitivityDecrease);
    sens = sens.mul(tf.pow(oneSubDec, spikeCounter));
    //console.log(spikeCounter.mean().dataSync(), "counter")
    sens = sens.add(this.sensitivityIncrease);
    this.sensitivity.assign(sens);

    //r-stdp
    //decay spike traces
    var spikeDecay = this.spikeDecay;
    spikeDecay = spikeDecay.mul(1 - this.spikeDecayRate);
    //update spike traces
    spikeDecay = tf.where(this.spikes, tf.onesLike(this.spikes).asType("float32"), spikeDecay);
    this.spikeDecay.assign(spikeDecay);

    U.for3d(-ks, -ks, -ks, ks+1, ks+1, ks+1, (x, y, z) =>
    {
      //shift spikes to be local to neuron
      var spikeDecayShift = U.shift(this.spikeDecay, [x, y, z]);
      //calculate difference between spike "timings"
      var spikeDiff = spikeDecayShift.sub(this.spikeDecay); //TODO: correct direction?
      spikeDiff = spikeDiff.mul(this.traceDir);
      //decay traces
      var trace = this.traces[[x, y, z]];
      trace = trace.mul(1 - this.traceDecayRate);
      //add spike diff to traces
      trace = trace.add(spikeDiff);
      this.traces[[x, y, z]].assign(trace);
    });
  }

  train(reward)
  {
    var ks = this.kernelSize;
    U.for3d(-ks, -ks, -ks, ks+1, ks+1, ks+1, (x, y, z) =>
    {
      var trace = this.traces[[x, y, z]];
      var syn = this.synapses[[x, y, z]];
      syn = syn.add(trace.mul(reward * this.learningRate));
      syn = syn.clipByValue(this.synMin, this.synMax);
      this.synapses[[x, y, z]].assign(syn);
    });
  }
}
