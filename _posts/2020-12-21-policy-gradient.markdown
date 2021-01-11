---
layout: post
title: "Introduction to Policy Gradient Method in Reinforcement Learning"
categories: reinforcement_learning
---
{% include math.html %}

Policy gradient is a reinforcement learning (RL) algorithm that directly tries to maximize the rewards (as opposed to using another proxy value, e.g., q function). Let's revisit the goal of reinforcement learning: it's to maximize the expected cumulative rewards, i.e.,

$$
p_\theta (\tau) = p(s_1)\prod^T_{t=1}\pi_\theta(a_t|s_t)p(s_{t+1}|s_t,a_t) \\
\mathop{\operatorname{arg\,max_\theta}} E_{\tau\sim p_\theta(\tau)} \sum_t r(s_t, a_t)
$$

We will try to find the best policy by taking the gradient of the objective

$$
J(\theta) =  E_{\tau\sim p_\theta(\tau)} \sum_t r(s_t, a_t)
$$

We will first go through the varient of policy gradient method called REINFORCE; it uses Monte-Carlo methods in that it learns from observations without modeling the dynamics and uses the mean of observations as the approximation of the expected return.

Recall that the typical steps of training in reinforcement learning consists of repeating 1) generating samples, 2)fitting a model, 3)improving the policy. We can estimate $$J(\theta)$$ by running the policy and observing the rewards. For example, if I'm using reinforcement learning to train a polcy plaing an Atari game, I would start by making random moves, get a set of state and action pairs $$s_1, a_1, s_2, a_2$$. Our reward function will be the score displayed in the game, and can be mapped as $$r(s_i, a_i)$$. It would be a good idea to have a few or more trajectories before we fit a model to get more data; if we had $$N$$ trajectories, then

$$
J(\theta) = \frac{1}{N} \sum_i^N \sum_t^T r(s_{i,t}, a_{i,t})
$$

After some math jugling, you will find the gradient is

$$
\triangledown_\theta J(\theta) = \sum_i \bigl( \sum_t \triangledown_\theta \log \pi_{\theta} (a_{i,t} \vert s_{i,t}) \bigr) \bigl( \sum_t r(s_{i,t}, a_{i,t}) \bigr)
$$

So you calculate the gradient and update the policy by
$$
\theta_{t+1} \leftarrow \theta_t + \alpha \triangledown J(\theta_t)
$$

Here's a sample pseudocode snippet in pytorch:
```python
import torch
from torch import nn
from torch import optim

def batch_train(observation_dim, action_dim, hidden_layer_dim, learning_rate):

    policy = nn.Sequential(nn.Linear(observation_dim, hidden_layer_dim),
                            nn.Linear(hidden_layer_dim, action_dim))
    optimizer = optim.Adam(policy.parameters(), learning_rate)

    for _ in range(NUM_STEPS):
        observations, actions, rewards, next_observations = sample()
        n = observations.shape[0]
        acts_dist_given_obs = policy(observations)
        loss = -torch.mean(torch.mul(acts_dist_given_obs.log_prob(actions), torch.sum(rewards, 1)))
        optimizer.zero_grad()
        loss.backwards()
        optimizer.step()
```

The policy gradient is nice because we don't really need to know about distributions of states or the dynamics model $$p(s_{t+1} \vert s_t,a_t)$$. It's also useful when your action space is infinite, because you can actually take the gradient to move towards a better policy. Unlike in policy gradient, in policy iteration methods, the "improving the policy" step will have to iterate through all the possible actions to find one that maximizes the q-value (expected rewards given a state and an action).

The downside is that we need a lot of samples to deal with the variance. Hihg variance means the policy is going to not converge well and we will spend a lot of time training. However, without sampling more, there are some tricks to help reduce the variance.

### Reducing variance

#### Using Rewards-To-Go
Take a look at our policy gradient.

$$
\triangledown_\theta J(\theta) = \sum_i \bigl( \sum_{t=1}^T \triangledown_\theta \log \pi_{\theta} (a_{i,t} \vert s_{i,t}) \bigr) \bigl( \sum_{t=1}^T r(s_{i,t}, a_{i,t}) \bigr)
$$

We know that an action taken at $$t'$$ cannot affect rewards at $$t < t'$$. So we can skip adding it to our gradient. By virtue of having fewer items to sum, we can reduce variance. Witih this modification, our gradient becomes

$$
\triangledown_\theta J(\theta) = \sum_i \bigl( \sum_t \triangledown_\theta \log \pi_{\theta} (a_{i,t} \vert s_{i,t}) \bigr) \bigl( \sum_{t'=t}^T r(s_{i,t'}, a_{i,t'}) \bigr)
$$

In our code, we will use `rewards_to_go` instead of `torch.sum(rewards, 1)`
```python
rewards_to_go = torch.zeros(rewards.shape[0])
for t in range(rewards.shape[1]):
    rewards_to_go[t] = torch.sum(rewards[:, t:])
```

#### Using Baseline
Another idea is to subtract a baseline. This will reduce variance, while keeping the bias unchanged because
$$
E_{a_t ~ \pi(\theta)}\bigl[ \triangledown_\theta \log \pi_\theta(a_t \vert s_t)b(s_t) \bigr] = 0
$$.

We could use the average rewards $$b = \frac{1}{N}\sum_i^N \sum_{t'=t}^T r(s_t', a_t')$$ from the policy $$\pi$$, or train a simple neural net that predicts rewards based on observation. Our gradient will now be

$$
\triangledown_\theta J(\theta) = \sum_i \bigl( \sum_t \triangledown_\theta \log \pi_{\theta} (a_{i,t} \vert s_{i,t}) \bigr) \bigl( \sum_{t'=t}^T r(s_{i,t'}, a_{i,t'}) \bigr)
$$

These tricks will be a good motivator for the Actor-Critic method, and I'll discuss that next time!

{::comment}
// how do we improve variance? what if good samples have 0?
// how to reduce variance? use rewards to go, subtract a baseline (average reward)
https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html
https://towardsdatascience.com/policy-gradients-in-a-nutshell-8b72f9743c5d
https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html
https://lilianweng.github.io/lil-log/2018/02/19/a-long-peek-into-reinforcement-learning.html#monte-carlo-methods
{:/comment}
