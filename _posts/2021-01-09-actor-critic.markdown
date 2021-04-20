---
layout: post
title: "Intro to Actor-Critic Method"
categories: reinforcement_learning
---
{% include math.html %}

The actor-critic method is one of the methods in reinforcement learning (RL), in which we learn both the value function and the policy. In the policy gradient method, we learned only the policy and relied on a multitude of observations for reward estimation. In Q-learning (to be discussed in the later post), we estimate only the reward, while keeping the policy.

I will discuss the actor-critic method by comparing it against the policy gradient method (discussed in this [post]({% post_url 2020-12-20-reinforcement-learning-primer-rewards %})), so getting a bit of refresher on policy gradient will make this post easier to understand!

The policy gradient method often suffers from high variance during training. The actor-critic method attempts to reduce the variance by fitting functions that are components of the true objective. But first, let's define a couple of functions.

## Q-function

**Q-function ($$Q^\pi(s_t, a_t)$$)** provides the expected cumulative reward from time $$t$$ when taking action $$a_t$$ in state $$s_t$$ as you roll out the future action-state pairs according to policy $$\pi$$.

$$
Q^\pi(s_t, a_t) = \sum_{t'=t}^T E_{\pi_\theta} [r(s_{t'}, a_{t'})|s_t, a_t]
$$

The true RL objective is to maximize the expected total rewards, and can be rewritten in terms of Q function as

$$
E_{\tau \sim p_\theta(\tau)} \Big[ \sum_{t=1}^T r(s_t, a_t) \Big] \\
= E_{s_1 \sim p(s_1)} [E_{a_1 \sim \pi(a_1 \vert s_1)}[Q(s_1, a_1) \vert s_1]]
$$

## Value function

**Value function ($$V(s_t)$$)** provides the expected cumulative reward from $$s_t$$.

$$
V^\pi(s_t, a_t) = \sum_{t'=t}^T E_{\pi_\theta} [r(s_t', a_t') | s_t]
$$

The RL objective can be rewritten in terms of $$V$$ is

$$E_{s_1 \sim p(s_1)}[V^\pi(s_1)]$$

From the earlier equality mentioned in Q-function, we can also see

$$V^\pi(s) = E_{a_t \sim \pi_\theta (a_t \vert s_t)}[Q(s_t, a_t)]$$

### Actor-Critic

Recall in the policy gradient variant with rewards-to-go and baseline, the gradient that we used to update the policy parameters is

$$
\triangledown_\theta J(\theta) = \frac{1}{N} \sum_i \bigl( \sum_t \triangledown_\theta \log \pi_{\theta} (a_{i,t} \vert s_{i,t}) \bigr) \bigl( \sum_{t'=t}^T r(s_{i,t'}, a_{i,t'}) -  \frac{1}{N}\sum_i^N \sum_{t'=t}^T r(s_{i,t'}, a_{i,t'}) \bigr)
$$

Now that you know the definition of $$Q^\pi$$ and $$V^\pi$$, you should be able to see the expected values of the last terms are $$Q^\pi$$ and $$V^\pi$$, respectively.

$$
Q^\pi(s_t, a_t) = E_{\pi_\theta} (\sum_{t'=t}^T r(s_{i,t'}, a_{i,t'})) \\
V^\pi(s_t) = E_{\pi_\theta} (\frac{1}{N}\sum_i^N \sum_{t'=t}^T r(s_{i,t'}, a_{i,t'}))
$$

The RL objective can be rewritten as

$$
\triangledown_\theta J(\theta) = \frac{1}{N} \sum_i \bigl( \sum_t \triangledown_\theta \log \pi_{\theta} (a_{i,t} \vert s_{i,t}) \bigr) \bigl( Q^\pi(s_t,a_t) - V^\pi(s_t) \bigr)
$$

We call $$A^\pi(s_t, a_t) = Q^\pi(s_t,a_t) - V^\pi(s_t)$$ **advantage**, a measure of how good $$a_t$$ is in terms of maximizing rewards relative to the average in that state.

Recall Q-function is the total rewards from taking action $$a_t$$ in state $$s_t$$. We directly know the reward at $$t$$; we just call $$r(a_t, s_t)$$, but from $$t+1$$, the expected reward should be $$V^\pi(s_{t+1})$$. Therefore, advantage function can be rewritten as

$$
A^\pi(a_t,s_t) = r(a_t, s_t) + V^\pi(s_{t+1}) - V^\pi(s_t)
$$

Now there's only one value function in our RL objective. That's great news, because we can fit  just one value model to predict $$V^\pi$$. Having a model for $$V^\pi$$ should give a smoother gradient to $$J(\theta)$$. Our value target is approximated as $$\frac{1}{N} \sum_i^N \sum_t^T r(a_{i,t}, s_{i,t})$$. (Remember we used this value directly in $$J(\theta$$ in policy gradient? We are now using a model instead of using the value directly.) We can sample the policy to get pairs of $$(s_t,  \sum_t^T r(a_t, s_t))$$. Since the value function is equivalent to the expected value of q-values, we can use $$y_{i,t}$$ in estimating $$\hat{V}$$. The objective function in estimating $$V^\theta$$ is then

$$
\left\lVert \hat{V}(s_t) - \sum_{t'=t}^T r\right\rVert^2
$$

During the actor-critic method, we switch between fitting on this value function ($$V^\pi$$), a.k.a., critic, and improving the policy $$\pi$$, a.k.a., actor. The method name makes sense; the value function tells you the expected rewards, effectively "criticizing" your policy, while the policy is one that determines what actions to take, thereby being an "actor" that acts upon the critic's criticism. During the training process, we switch back and forth between improving the critic AND improving the actor, and each one of their improvement should further help the other.

The steps in each training cycle is

1. generate sample pairs $$(s_i, a_i)$$ from $$\pi_\theta$$ and record the rewards
2. fit the value function $$\hat{V}_\phi^\pi(s)$$ to the sum of rewards to go (q-values)
3. estimate advantage $$\hat{A}^\pi(s_i, a_i) = r(a_t, s_t) + V^\pi(s_{t+1}) - V^\pi(s_t)$$
4. calculate the gradient of the RL objective $$\triangledown_\theta J(\theta) = \sum_i \triangledown_\theta \log \pi_\theta(a_i\vert s_i) \hat{A}^\pi(s_i, a_i)$$
5. Update the parameters of the model $$\theta \leftarrow \theta + \alpha \triangledown_\theta J(\theta)$$

What we've discussed so far in pseudo-code is
```python
import torch
from torch import nn
from torch import optim

def sample(policy: nn.Module) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample based on the given policy."""
    ...

def batch_train(observation_dim: int, action_dim: int, hidden_layer_dim: int, learning_rate: float):
    actor = nn.Sequential(nn.Linear(observation_dim, hidden_layer_dim),
                            nn.Linear(hidden_layer_dim, action_dim))
    actor_optimizer = optim.Adam(actor.parameters(), learning_rate)
    critic = nn.Sequential(nn.Linear(observation_dim, hidden_layer_dim),
                            nn.Linear(hidden_layer_dim, 1))
    critic_optimizer = optim.Adam(critic.parameters(), learning_rate)

    for _ in range(NUM_STEPS):
        observations, actions, rewards, next_observations = sample(actor)
        q_values = rewards.flip(1).cumsum(1).flip(1)
        for _ in range(NUM_CRITIC_UPDATES_PER_STEP):
            q_values = rewards.flip(1).cumsum(1).flip(1)
            critic_rewards = critic(observations)
            critic_optimizer.zero_grad()
            critic_loss = nn.MSELoss(critic_rewards, q_values)
            critic_optimizer.zero_grad()
            critic_loss.backwards()

        advantage = rewards + critic(next_observations) - critic(observation)

        for _ in range(NUM_ACTOR_UPDATES_PER_STEP):
            acts_given_obs = actor(observations)
            actor_loss = -torch.mean(torch.mul(acts_given_obs.log_prob(actions), advantage)
            actor_optimizer.zero_grad()
            actor_loss.backwards()
            actor_optimizer.step()
```

To avoid repeating myself, refer to this [post]({% post_url 2020-12-20-reinforcement-learning-primer-rewards %}) for the more "verbose" description of the code to make it easier to follow along, in case you're not so familiar with pytorch.

#### Discount factor
One common thing to see in estimating $$V^\pi(s)$$ is discounting the future. The usefulness of this becomes more apparent when your time horizon infinite.

$$V^\pi(s) = E_{a \sim \pi} \big[\sum_{t=0}^\infty r(s_t, a_t \vert s_0 = s)\big] $$

In the above case, we will never finish finding out $$V^\pi$$. Introducing a discount rate $$\gamma$$, the value function now becomes

$$V^\pi(s) = E_{a \sim \pi} \big[\sum_{t=0}^\infty \gamma^t r(s_t, a_t \vert s_0 = s)\big] $$

where the rewards in the far future will be effectively zero. This also helps reducing variance even further because estimations in the farther future are less likely to be "correct".

The advantage will also discount the next observation by $$\gamma$$, i.e., $$\hat{A}^\pi(s_i, a_i) = r(a_t, s_t) + \gamma V^\pi(s_{t+1}) - V^\pi(s_t)$$

#### Pros and Cons
The actor-critic method provides lower variance than the policy gradient method and tends to converge better. It's more **sample-efficient** in that your model could converge with fewer samples. But note that our actor is always sampling based on its own policy; this means that it may not take actions that could be potentially very good but are drastically different from what it has tried. (Such a method is called an on-policy algorithm, meaning it samples according to its policy.) That means your actor can be great at converging into some local maximum, but can be stuck in it and **not find the global maximum**.

This leads well into Q-learning, which is an **off-policy** algorithm, another great approach in reinforcement learning.

