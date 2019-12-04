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

  diamond(x, y, z)
  {
    if(x == 0 && y == 0)
    {
      return true;
    }
    if(y == 0 && z == 0)
    {
      return true;
    }
    if(x == 0 && z == 0)
    {
      return true;
    }
    return false;
  }

  reset()
  {
    this.potential = tf.variable(tf.zeros(this.cubeShape));
    this.sensitivity = tf.variable(tf.zeros(this.cubeShape));
    //this.sensitivity = tf.variable(tf.ones(this.cubeShape));
    this.spikes = tf.variable(tf.zeros(this.cubeShape, "bool"));
    this.spikeDecay = tf.variable(tf.zeros(this.cubeShape));
    var synapses = {};
    var traces = {};
    var ks = this.kernelSize;
    //TODO: convert to a single tensor with reshapes n stuff
    U.for3d(-ks, -ks, -ks, ks+1, ks+1, ks+1, (x, y, z) =>
    {
      //if(!this.diamond(x, y, z)) { return; }

      //var w = tf.variable(tf.zeros(this.cubeShape));
      var w = tf.variable(tf.randomUniform(this.cubeShape, 0, 1));
      //var w = tf.variable(tf.ones(this.cubeShape));
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
    //this.spikes.assign(tf.greater(this.potential, threshold));
    this.spikes.assign(tf.greater(this.potential, tf.onesLike(this.potential).mul(0.02)));
    var potential = this.potential;
    //reset potential where spikes occur
    potential = tf.where(this.spikes, tf.zerosLike(potential), potential);
    //decay potential
    potential = potential.mul(1 - this.potentialLeakRate);

    var spikeCounter = tf.zeros(this.cubeShape);
    U.for3d(-ks, -ks, -ks, ks+1, ks+1, ks+1, (x, y, z) =>
    {
      //if(!this.diamond(x, y, z)) { return; }

      //shift spikes to be local to neuron
      var spikeShift = U.shift(this.spikes, [x, y, z]).asType("float32");
      //count spikes for sensitivity calculation later
      spikeCounter = spikeCounter.add(spikeShift);
      //add to potential
      //var spikeSynSens = this.spikes.asType("float32").mul(this.synapses[[x, y, z]]).mul(this.sensitivity);
      var spikeSynSens = spikeShift.mul(this.synapses[[x, y, z]]).mul(this.sensitivity);
      potential = potential.add(spikeSynSens);
    });

    //increase potential slowly over time
    potential = potential.add(this.sensitivity.mul(this.potentialIncrease));
    this.potential.assign(potential);

    //update sensitivity
    var sens = this.sensitivity
    var oneSubDec = tf.onesLike(spikeCounter).sub(this.sensitivityDecrease);
    sens = sens.mul(tf.pow(oneSubDec, spikeCounter));
    //console.log(spikeCounter.mean().dataSync(), "counter")
    sens = sens.add(this.sensitivityIncrease);
    this.sensitivity.assign(sens);

    //r-stdp
    U.for3d(-ks, -ks, -ks, ks+1, ks+1, ks+1, (x, y, z) =>
    {
      //if(!this.diamond(x, y, z)) { return; }

      //shift spikes to be local to neuron
      var spikeDecayShift = U.shift(this.spikeDecay, [x, y, z]);
      var spikeShift = U.shift(this.spikes, [x, y, z]);
      //decay where spikes occur, else 0
      spikeDecayShift = tf.where(spikeShift, spikeDecayShift, tf.zerosLike(spikeDecayShift));

      //var spikeDiff = spikeDecayShift.sub(this.spikeDecay); //TODO: correct direction?
      var decay = this.spikeDecay;
      //decay where spikes occur, else 0
      decay = tf.where(this.spikes, decay, tf.zerosLike(decay));

      //calculate difference between spike "timings"
      var spikeDiff = decay.sub(spikeDecayShift); //TODO: correct direction?
      spikeDiff = spikeDiff.sign().sub(spikeDiff);
      spikeDiff = spikeDiff.mul(this.traceDir);
      //decay traces
      var trace = this.traces[[x, y, z]];
      trace = trace.mul(1 - this.traceDecayRate);
      //add spike diff to traces
      trace = trace.add(spikeDiff);
      this.traces[[x, y, z]].assign(trace);
    });

    //decay spike traces
    var spikeDecay = this.spikeDecay;
    spikeDecay = spikeDecay.mul(1 - this.spikeDecayRate);
    //update spike traces
    spikeDecay = tf.where(this.spikes, tf.onesLike(this.spikes).asType("float32"), spikeDecay);
    this.spikeDecay.assign(spikeDecay);
  }

  train(reward)
  {
    var ks = this.kernelSize;
    U.for3d(-ks, -ks, -ks, ks+1, ks+1, ks+1, (x, y, z) =>
    {
      //if(!this.diamond(x, y, z)) { return; }
      var trace = this.traces[[x, y, z]];
      var syn = this.synapses[[x, y, z]];
      syn = syn.add(trace.mul(reward * this.learningRate));
      syn = syn.clipByValue(this.synMin, this.synMax);
      this.synapses[[x, y, z]].assign(syn);
    });
  }
}
