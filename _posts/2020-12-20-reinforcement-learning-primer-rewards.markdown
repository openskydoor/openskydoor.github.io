---
layout: post
title: "Reinforcement Learning Rewards-based Algorithms - Primer"
categories: reinforcement_learning
---

Reinforcement learning (RL) helps discover how agents ought to take actions (policy) in an environment (transition operator) in order to maximize the notion of cumulative reward. In this post, I'm going to discuss how we can define this problem more concretely with a reward function and the different objective functions we can optimize.

// say some example
As a refresher, we

state, action, policy, reward

We learned that we can use Markov decision process (MDP) to describe a reinforcement learning problem.

// diagram

where

$$
S: state space
A: action space
O: observation space
\epsilon: emission probability (p(s_t|o_t))
\Tau: transition operator (p(s_{t+1}|s_{t},a_{t}))
r: reward function (r(s,a))
$$


## Training Cycle
Let's call the parameters of the policy model ![](https://latex.codecogs.com/svg.latex?\theta), and [](https://latex.codecogs.com/svg.latex\pi_{\theta}) represents the policy with the parameters ![](https://latex.codecogs.com/svg.latex?\theta). Let's suppose we took actions according to the policy $\pi_{\theta}$ (rollout), and we got a sequence of states and action pairs, ![](https://latex.codecogs.com/svg.latex?s_0, a*0, s_1, a_1, ...). The goal is to find ![](https://latex.codecogs.com/svg.latex?\theta) such as that we maximize the expected rewards, i.e.,
![](https://latex.codecogs.com/svg.latex?arg&space;max*\theta&space;E_{\tau~p\*{\theta}(\tau)[\sum_t&space;r(s_t, a_t)]}), where ![](https://latex.codecogs.com/svg.latex?\tau) is the rollout ![](https://latex.codecogs.com/svg.latex?s_1,a_1,s_2,a_2,...) with a policy ![](https://latex.codecogs.com/svg.latex?\pi).

In supervised learning, we have inputs, labels, and model prediction, and the goal is to come up with the model that generates predictions that are as close to labels as possible, without losing the ability to generalize.
In reinforcement learning, your policy (![](https://latex.codecogs.com/svg.latex?\pi), which action to take) and model (state transition, ![](https://latex.codecogs.com/svg.latex?p(s_{t+1}|s_t,a_t) ) ) is responsible for generating the future states that it predicts on. Therefore, we can't simply do a single pass training (there is ample research going on in training reinforcement learning this way, called offline reinforcement learning) and we have a cycle:

1. generate samples ![](https://latex.codecogs.com/svg.latex?(s,a)). You run the policy or have some expert collected samples for bootstrapping.
2. fit a model (![](https://latex.codecogs.com/svg.latex?p(s_{t+1}\|s_t,a_t))) or estimate the total rewards (![](https://latex.codecogs.com/svg.latex?E(r)))
3. improve the policy
4. repeat!

There are a few options you can use in step 2 and 3. The first distinctions we will make are model-based vs. model-free.

## Model-based vs Model-free Learning

Going back to the markov diagram, we can say the probability of seeing a particular rollout is ![](https://latex.codecogs.com/svg.latex?p(s_1)\prod^T*{t=1}\pi*\theta(a_t%7Cs_t)p(s*{t+1}%7Cs_t,a_t)).


In the model-free approach, we do not try to learn ![](https://latex.codecogs.com/svg.latex?p(s_{t+1}%7Cs_t,a_t)).

In the model-based approach, we try to learn the system dynamics. For example, suppose we are building a robot that can pick up an object from one place and drop it in another place; in model-based reinforcement learning, we will have an understanding of where the robot arm will be (![](https://latex.codecogs.com/svg.latex?s_{t+1})) after it moved its arm right once (![](https://latex.codecogs.com/svg.latex?a_t)) from where it was (![](https://latex.codecogs.com/svg.latex?s_t))

In the next few posts, we will discuss a few options within model-free approach, starting from policy gradient.
