---
layout: post
title: "Intro to Policy Gradient Method"
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

We will first go through the variant of policy gradient method called REINFORCE; it uses Monte-Carlo methods in that it learns from observations without modeling the dynamics and uses the mean of observations as the approximation of the expected return.

Recall that the typical steps of training in reinforcement learning consists of repeating 1) generating samples, 2) fitting a model, 3) improving the policy. We can estimate $$J(\theta)$$ by running the policy and observing the rewards. For example, if I'm using reinforcement learning to train a policy playing an Atari game, I would start by making random moves, get a set of state and action pairs $$s_1, a_1, s_2, a_2$$. Our reward function will be the score displayed in the game, and can be mapped as $$r(s_i, a_i)$$. It would be a good idea to have a few or more trajectories before we fit a model to get more data; if we had $$N$$ trajectories, then

$$
J(\theta) = \frac{1}{N} \sum_i^N \sum_t^T r(s_{i,t}, a_{i,t})
$$

After some math juggling (if you want to know what this juggling is, check ![this](https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html#proof-of-policy-gradient-theorem)), you will find the gradient is

$$
\triangledown_\theta J(\theta) = \sum_i \bigl( \sum_t \triangledown_\theta \log \pi_{\theta} (a_{i,t} \vert s_{i,t}) \bigr) \bigl( \sum_t r(s_{i,t}, a_{i,t}) \bigr)
$$

So you calculate the gradient and update the policy by
$$
\theta \leftarrow \theta + \alpha \triangledown J(\theta)
$$

Here's a sample pseudo-code snippet assuming our policy has a neural network architecture implemented in pytorch:
```python
import torch
from torch import nn
from torch import optim

def sample(policy: nn.Module) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample based on the given policy."""
    ...

def batch_train(observation_dim: int, action_dim: int, hidden_layer_dim: int, learning_rate: float):
    policy = nn.Sequential(nn.Linear(observation_dim, hidden_layer_dim),
                            nn.Linear(hidden_layer_dim, action_dim))
    optimizer = optim.Adam(policy.parameters(), learning_rate)

    for _ in range(NUM_STEPS):
        observations, actions, rewards = sample(policy)
        acts_given_obs = policy(observations)
        total_rewards = torch.sum(rewards, 1).repeat(observations.shape[0], 1)
        loss = -torch.mean(torch.mul(acts_given_obs.log_prob(actions), total_rewards))
        optimizer.zero_grad()
        loss.backwards()
        optimizer.step()
```

Note I used the variable name `observations` here instead of `states`; in most realistic scenarios, getting a truthful state is hard -- but our observation is a good approximation of state. For example, an image is in two-dimensional space, but we can still guess the depth of different objects in the image pretty accurately, as if we were seeing them in real life.

Remember the trajectory is the state-action pairs of a policy rollout? Suppose you have a policy that plays an video game, and you play the game according to the policy continuously for $$T$$ time units. You decide to do $$n$$ such rollouts of the policy, and `sample` returns the relevant data for your policy training. It returns `observations` ($$n \times T \times \text{observation_dim} $$), `actions` ($$n \times T \times \text{action_dim}$$), and `rewards` that was added at each timestamp ($$n \times T$$). Each observation dimension could represent different things like whether you see an enemy in the screen, where your current position is in the game, etc. Each action could be different potential things you can do, move right, move left, shoot, or some continuous value that represents the angle you face.

Skip this paragraph if you're familiar with pytorch. In `batch_train` function, line 1 is defining the structure of the policy model; in our case, it's a very simple one with one hidden layer. Its input is observation, output is action probability distribution; this matches our intuition because our policy should tell us what to do based on what we see. The next one is initializing an Adam optimizer we're going to use, and the learning rate is equivalent of $$\alpha$$ in the the formula above. Another quirk about pytorch is calling the object is how you get the output of your model.. so `policy(observations)` returns $$\pi(a \vert s)$$. The last three lines are executing backpropagation to get the gradient and updating the parameters of the policy $$\theta$$.

#### Pros
Policy gradient is nice because we don't really need to know about the distribution of states or the dynamics model $$p(s_{t+1} \vert s_t,a_t)$$. It's also useful when your action space is infinite. In policy iteration methods, the "improving the policy" step will have to iterate through all possible actions in action space to find one that maximizes the q-value (expected rewards given a state and an action). However, in the policy gradient method, we have a gradient we can optimize, meaning we can sample a bit and still move towards a better policy.

#### Cons
The downside is that we still need a lot of samples to deal with the variance. High variance means the policy is going to not converge well and we will be spending a lot of time training. However, without sampling more, there are some tricks to help reduce the variance.

### Reducing variance

#### Using Rewards-To-Go
Take a look at our policy gradient.

$$
\triangledown_\theta J(\theta) = \sum_i \bigl( \sum_{t=1}^T \triangledown_\theta \log \pi_{\theta} (a_{i,t} \vert s_{i,t}) \bigr) \bigl( \sum_{t=1}^T r(s_{i,t}, a_{i,t}) \bigr)
$$

We know that an action taken at $$t'$$ cannot affect rewards at $$t < t'$$. So we can skip adding it to our gradient. By virtue of having fewer items to sum, we can reduce variance. Witih this modification, our gradient becomes

$$
\triangledown_\theta J(\theta) = \sum_i \bigl( \sum_{t=1}^T \triangledown_\theta \log \pi_{\theta} (a_{i,t} \vert s_{i,t}) \bigr) \bigl( \sum_{t'=t}^T r(s_{i,t'}, a_{i,t'}) \bigr)
$$

In our code, we will use `rewards_to_go` instead of `total_rewards`
```python
rewards_to_go = rewards.flip(1).cumsum(1).flip(1)
```

#### Using Baseline
Another idea is to subtract a baseline. This will reduce variance, while keeping the bias unchanged because
$$
E_{a_t ~ \pi(\theta)}\bigl[ \triangledown_\theta \log \pi_\theta(a_t \vert s_t)b(s_t) \bigr] = 0
$$.

We could use the average rewards as $$b = \frac{1}{N}\sum_i^N \sum_{t'=t}^T r(s_t', a_t')$$ as we observe while rolling out the policy $$\pi$$. Our gradient will now be

$$
\triangledown_\theta J(\theta) = \sum_i \bigl( \sum_{t=1)^T \triangledown_\theta \log \pi_{\theta} (a_{i,t} \vert s_{i,t}) \bigr) \bigl( \sum_{t'=t}^T r(s_{i,t'}, a_{i,t'}) - b\bigr)
$$

What this means in practice is that our policy changes by how good the current policy is relative to the expected rewards in that particular state, instead of just by how much the rewards are.

These tricks will be a good motivator for the Actor-Critic method, and I'll discuss that next time!

{::comment}
// how do we improve variance? what if good samples have 0?
// how to reduce variance? use rewards to go, subtract a baseline (average reward)
https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html
https://towardsdatascience.com/policy-gradients-in-a-nutshell-8b72f9743c5d
https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html
https://lilianweng.github.io/lil-log/2018/02/19/a-long-peek-into-reinforcement-learning.html#monte-carlo-methods
{:/comment}
