
//<![CDATA[

// a few things don't have var in front of them - they update already existing variables the game needs
// These are probably things I want to tweak a bit, since it's only looking right in front of it now
lanesSide = 0;
patchesAhead = 1;
patchesBehind = 0;
trainIterations = 10000;

// the number of other autonomous vehicles controlled by your network
// I get the average of these for speed, but they can't collaborate
otherAgents = 0; // max of 9

// These don't need changed, except perhaps temporal_window
var num_inputs = (lanesSide * 2 + 1) * (patchesAhead + patchesBehind);
var num_actions = 5;
var temporal_window = 3;
var network_size = num_inputs * temporal_window + num_actions * temporal_window + num_inputs;

// This is just the input layer and doesn't need changed
var layer_defs = [];
    layer_defs.push({
    type: 'input',
    out_sx: 1,
    out_sy: 1,
    out_depth: network_size
});

// Just an initial layer so they show how it works. This is something I need to change!
layer_defs.push({
    type: 'fc',
    num_neurons: 1,
    activation: 'relu'
});

// Final regression layer; fine as-is likely
layer_defs.push({
    type: 'regression',
    num_neurons: num_actions
});

// These are more advanced tweaks. I bet I can do some hyperparams tuning here. 
// More at: https://github.com/karpathy/convnetjs/blob/master/build/deepqlearn.js
var tdtrainer_options = {
    learning_rate: 0.001, // Tweak this
    momentum: 0.0,  // Tweak this
    batch_size: 64, // Tweak this
    l2_decay: 0.01  // Re-read on l2 
};


// These are additional hyperparams that I can tweak
var opt = {};
opt.temporal_window = temporal_window;
opt.experience_size = 3000;
opt.start_learn_threshold = 500;
opt.gamma = 0.7;
opt.learning_steps_total = 10000;
opt.learning_steps_burnin = 1000;
opt.epsilon_min = 0.0;
opt.epsilon_test_time = 0.0;
opt.layer_defs = layer_defs;
opt.tdtrainer_options = tdtrainer_options;

// This creates the brain and runs it
brain = new deepqlearn.Brain(num_inputs, num_actions, opt);

learn = function (state, lastReward) {
  brain.backward(lastReward);
  var action = brain.forward(state);
  draw_net();
  draw_stats();
  return action;
}

//]]>
