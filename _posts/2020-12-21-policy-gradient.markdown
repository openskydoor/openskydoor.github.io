---
layout: post
title: "Introduction to Policy Gradient Method in Reinforcement Learning"
categories: reinforcement_learning
---
{% include math.html %}

Policy gradient is a reinforcement learning (RL) algorithm that directly tries to maximize the rewards (as opposed to using another proxy value, e.g., q function). Let's revisit the goal of reinforcement learning: it's to maximize the expected cumulative rewards, i.e.,
{% katex display %}
p_\theta (\tau) = p(s_1)\prod^T_{t=1}\pi_\theta(a_t|s_t)p(s_{t+1}|s_t,a_t) \\
\mathop{\operatorname{arg\,max_\theta}} E_{\tau\sim p_\theta(\tau)} \sum_t r(s_t, a_t)
{% endkatex %}

We will try to find the best policy by taking the gradient of the objective {% katex %}J(\theta) =  E_{\tau\sim p_\theta(\tau)} \sum_t r(s_t, a_t) {% endkatex %}.

In this post, we will go through the varient of policy gradient method called REINFORCE; it uses Monte-Carlo methods in that it learns from observations without modeling the dynamics and uses the mean of observations as the approximation of the expected return

Recall that the typical steps of training in reinforcement learning consists of repeating 1) generating samples, 2)fitting a model, 3)improving the policy. We can estimate {% katex %}J(\theta){% endkatex %} by running the policy and observing the rewards. For example, if I'm using reinforcement learning to train a polcy plaing an Atari game, I would start by making random moves, get a set of state and action pairs {% katex %}s_1, a_1, s_2, a_2{% endkatex %}. Our reward function will be the score displayed in the game, and can be mapped as {% katex %}r(s_i, a_i){% endkatex %}. It would be a good idea to have a few or more trajectories before we fit a model to get more data; if we had {% katex %}N{% endkatex %} trajectories, then
{% katex display %}
J(\theta) = \frac{1}{N} \sum_i^N \sum_t^T r(s_{i,t}, a_{i,t})
{% endkatex %}

After some math jugling, you will find the gradient is
{% katex display %}
\triangledown_\theta J(\theta) = \sum_i (\sum_t \triangledown_\theta \log \pi_{\theta})
{% endkatex %}

So you calculate the gradient and update the policy by
{% katex display %}
\theta_{t+1} \leftarrow \theta_t + \alpha \triangledown J(\theta_t)
{% endkatex %}

The policy gradient is nice because we don't really need to know about distributions of states or the dynamics model {% katex %}p(s_{t+1}|s_t,a_t){% endkatex %}. It's also useful when your action space is infinite, because you can actually take the gradient to move towards a better policy. Unlike in policy gradient, in policy iteration methods, the "improving the policy" step will have to iterate through all the possible actions to find one that maximizes the q-value (expected rewards given a state and an action).

The downside is that we need a lot of samples to deal with the variance. How to deal with reducing the variance will be discussed in a separate post.

// how do we improve variance? what if good samples have 0?
// how to reduce variance? use rewards to go, subtract a baseline (average reward)

https://towardsdatascience.com/policy-gradients-in-a-nutshell-8b72f9743c5d

{% bibliography --cited %}
