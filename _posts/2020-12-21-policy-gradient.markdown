---
layout: post
title: "Intro to Policy Gradient Method"
categories: reinforcement_learning
---
{% include math.html %}

Policy gradient is a reinforcement learning (RL) algorithm that directly tries to maximize the rewards (as opposed to using another proxy value, e.g., q function). Let's revisit the goal of reinforcement learning: it's to maximize the expected cumulative rewards, i.e.,

$$
p_\theta (\tau) = p(s_1)\prod^T_{t=1}\pi_\theta(a_t|s_t)p(s_{t+1}|s_t,a_t),  \\
\text{where } \tau = {s_1, a_1, s_2, a_2, ...} \\
\mathop{\operatorname{arg\,max_\theta}} E_{\tau\sim p_\theta(\tau)} \sum_t^T r(s_t, a_t)
$$

We have an objective function, which means we can use its gradient and use it to improve our policy:

$$
J(\theta) =  E_{\tau\sim p_\theta(\tau)} \sum_t r(s_t, a_t) \\
\theta \leftarrow \theta + \alpha \triangledown_\theta J(\theta)
$$

This is the gist of policy gradient.

We will first go through a variant of policy gradient method called REINFORCE.

Recall that a typical RL method consists of repeating the following steps: 1) generating samples, 2) fitting a dynamics model/estimating the return, 3) improving the policy.

In REINFORCE, for step (2), it uses the Monte-Carlo method; it uses the mean of the observations as the approximation of the expected return. For example, if I'm using reinforcement learning to train a policy playing an Atari game, I would start by making random moves, get a set of state and action pairs $$s_1, a_1, s_2, a_2$$. Our reward function will be the score displayed in the game, and can be mapped as $$r(s_i, a_i)$$. Since we're using the mean as our approximation, having many samples should give us higher confidence, so it would be a good idea to have at least a few trajectories before we move onto step (3); if we had $$N$$ trajectories, then

$$
J(\theta) = \frac{1}{N} \sum_i^N \sum_t^T r(s_{i,t}, a_{i,t})
$$

After some math juggling (if you want to know what this juggling is, ![this](https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html#proof-of-policy-gradient-theorem) is a good post), you will find the gradient is

$$
\triangledown_\theta J(\theta) = \sum_i \bigl( \sum_t \triangledown_\theta \log \pi_{\theta} (a_{i,t} \vert s_{i,t}) \bigr) \bigl( \sum_t r(s_{i,t}, a_{i,t}) \bigr)
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

Note I used the variable name `observations` here instead of `states`; in most realistic scenarios, getting a truthful state is hard -- but our observation is a good approximation of the state. For example, an image is in two-dimensional space, but we can still guess the depth of different objects in the image pretty accurately, as if we were seeing them in real life. Of course there's some uncertainty involved in relying on observations alone, but for now it's a topic for another time.

Let's decompose the code for a bit. Remember a trajectory is state-action pairs from a policy rollout? Suppose you have a policy that plays an video game, and you play the game according to the policy continuously for $$T$$ time units. You decide to do $$n$$ such rollouts of the policy. The `sample` function is proxy of this step and returns these relevant data: `observations` ($$n \times T \times \text{observation_dim} $$), `actions` ($$n \times T \times \text{action_dim}$$), and `rewards` that was added at each timestamp ($$n \times T$$). Each observation dimension could represent different things like whether you see an enemy in the screen, where your current position is in the game, etc. Each action could represent different things you can do, move right, move left, shoot, or some continuous value that represents the angle you face.

Skip this paragraph if you're familiar with pytorch; think of it as `--verbose` mode in code reading! In `batch_train` function, the first defines the structure of the policy model; in our case, it's a very simple neural net with one hidden layer. Its input is observation, its output, action probability distribution; this matches our intuition because our policy should tell us what to do based on what we see. The next line initializes an Adam optimizer we're going to use; the learning rate is equivalent to $$\alpha$$ in the the formula above. Another quirk about pytorch is calling the object is how you get the output of your model.. so `policy(observations)` returns $$\pi(a \vert s)$$. The last three lines are executing routine backpropagation and parameter $$\theta$$ updates.

#### Pros
Policy gradient is great because we don't really need to know about the distribution of states or the dynamics model $$p(s_{t+1} \vert s_t,a_t)$$. It's also useful when your action space is infinite. There are some approaches in RL, such as Q-learning (to be discussed in a later post) that requires you to take all possible actions before you can take "improve the policy" step for best result. However, in the policy gradient method, we have a gradient we can optimize, meaning we can sample a bit and still inch towards a better policy.

#### Cons
The downside is that we still need a lot of samples to deal with the variance. Remember that we were generating samples and taking the mean for estimated rewards? If your sample size is small, your estimation will fluctuate a lot and your policy will be confused through the learning process... High variance means the policy is going to not converge well and we will be spending a lot of time training. However, without sampling more, there are some tricks to help reduce the variance.

### Reducing Variance

#### Using Rewards-To-Go
Notice the reward summation in the policy gradient:

$$
\triangledown_\theta J(\theta) = \sum_i \bigl( \sum_{t=1}^T \triangledown_\theta \log \pi_{\theta} (a_{i,t} \vert s_{i,t}) \bigr) \bigl( \sum_{t=1}^T r(s_{i,t}, a_{i,t}) \bigr)
$$

We know that an action taken at $$t'$$ cannot affect rewards at $$t < t'$$. So we can skip adding it to our gradient. By virtue of having fewer items to sum (fewer things that could change...), we can reduce variance. With this modification, our gradient becomes

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

We could use the average rewards as $$b = \frac{1}{N}\sum_i^N \sum_{t'=t}^T r(s_{i, t'}, a_{i, t'})$$, update it as we observe more policy rollouts. Our gradient will now be

$$
\triangledown_\theta J(\theta) = \sum_i \bigl( \sum_{t=1}^T \triangledown_\theta \log \pi_{\theta} (a_{i,t} \vert s_{i,t}) \bigr) \bigl( \sum_{t'=t}^T r(s_{i,t'}, a_{i,t'}) - b\bigr)
$$

What this means in practice is that our policy changes by how good the current policy is relative to the expected rewards in that particular state, instead of just by how much the rewards are.

These tricks are a good segue for the Actor-Critic method, and I'll discuss that next time!

{::comment}
// how do we improve variance? what if good samples have 0?
// how to reduce variance? use rewards to go, subtract a baseline (average reward)
https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html
https://towardsdatascience.com/policy-gradients-in-a-nutshell-8b72f9743c5d
https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html
https://lilianweng.github.io/lil-log/2018/02/19/a-long-peek-into-reinforcement-learning.html#monte-carlo-methods
{:/comment}
