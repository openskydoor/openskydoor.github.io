---
layout: post
title: "Actor-Critic Methods in Reinforcement Learning"
categories: reinforcement_learning
---
{% include math.html %}

In my previous [post]({% post_url 2020-12-20-reinforcement-learning-primer-rewards %}) on reinforcement learning (RL) based on rewards, I mentioned the true objective of reinforcement function is find the parameters of your policy such that you maximize the expected value of the total rewards, i.e., $$\mathop{\operatorname{arg\,max_\theta}} E_{\tau\sim p_\theta(\tau)} \sum_t r(s_t, a_t)$$. For example, you have come up with some architecture for your model, e.g., your neural net, and then during the training phase, you need to find the right numbers for the matrices (parameters). In each optimization step during the training, you need to update your parameters a bit by bit based on your objective function. Our true object has quite a few values to calculate, so computer scientists have come up with some other objective functions that are nicer to use. // list why value/q functions.


Let's define a couple of value functions.

## Q-function

*Q-function ($$Q(s_t, a_t)$$)* provides total reward from taking $$a_t$$ in $$s_t$$.

To help understanding the relationship of this to the RL objective, let me give you an example. If we knew the reward of taking action $$a_1$$ in state $$s_1$$, and understood the probability distribution of taking $$a_1$$ in $$s_1$$ and that of the state $$s_1$$, then, the RL objective can be rewritten as

$$E_{s_1 ~ p(s_1)} [E_{a_1 ~ \pi(a_1 \vert s_1)}[Q(s_1, a_1) | s_1]]$$.

This will be an important fact for motivating Q-learning later.

## Value function

*Value function ($$V(s_t)$$)* provides total reward from $$s_t$$. Note that it doesn't depend on the action, so that it should sum Q-values over all the possible actions per $$a_t ~ \pi(a_t|s_t)$$. The RL objective can be rewritten as

$$E_{s_1 ~ p(s_1)}[V(s_1)]$$.

### Actor-Critic

Recall in the policy gradient method (related [post]({% post_url 2020-12-21-policy-gradient %}), the gradient that we used to update the policy parameters is

$$


// talk about policy gradient
// high variance with small sample
// import with baseline, rewards to go
// estimate by fitting sampled reward sums
//
An actor-critic method is like a hybrid of using policy gradients and value functions.

Instead of fitting the RL objective directly, we fit either $$V(s)$$ or $$Q(s, a)$$, and update the

estimate value function or Q-function of the current policy,
use it to improve policy

