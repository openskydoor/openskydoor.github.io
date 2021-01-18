---
layout: post
title: "Intro to Actor-Critic Method"
categories: reinforcement_learning
---
{% include math.html %}

In my previous [post]({% post_url 2020-12-20-reinforcement-learning-primer-rewards %}) on reinforcement learning (RL) based on rewards, I mentioned the true objective of reinforcement function is find the parameters of your policy such that you maximize the expected value of the total rewards, i.e., $$\mathop{\operatorname{arg\,max_\theta}} E_{\tau\sim p_\theta(\tau)} \sum_t r(s_t, a_t)$$. For example, you have come up with some architecture for your model, e.g., your neural net, and then during the training phase, you need to find the right numbers for the matrices (parameters). In each optimization step during the training, you need to update your parameters a bit by bit based on your objective function. Our true object has quite a few values to calculate, so computer scientists have come up with some other objective functions that are nicer to use. // list why value/q functions.


Let's define a couple of value functions.

## Q-function

*Q-function ($$Q^\pi(s_t, a_t)$$)* provides total reward from taking $$a_t$$ in $$s_t$$ when you roll out the action-state pairs according to policy $$\pi$$.

To help understanding the relationship of this to the RL objective, let me give you an example. If we knew the reward of taking action $$a_1$$ in state $$s_1$$, and understood the probability distribution of taking $$a_1$$ in $$s_1$$ and that of the state $$s_1$$, then, the RL objective can be rewritten as

$$E_{s_1 \sim p(s_1)} [E_{a_1 \sim \pi(a_1 \vert s_1)}[Q(s_1, a_1) \vert s_1]]$$

This will be an important fact for motivating Q-learning later.

## Value function

*Value function ($$V(s_t)$$)* provides total reward from $$s_t$$. Note that it doesn't depend on the action, so that it should sum Q-values over all the possible actions per $$a_t \sim \pi(a_t \vert s_t)$$. The RL objective can be rewritten as

$$E_{s_1 \sim p(s_1)}[V^\pi(s_1)]$$

Note that your value function is an expected value of q-values.

$$V^\pi(s) = E_{a_t \sim \pi_\theta (a_t \vert s_t)}[Q(s_t, a_t)]$$

### Actor-Critic

The Actor-Critic method derives from the policy gradient, but we fit a model for the part of the gradient that is responsible for rewards so as to reduce the variance.

Recall in the policy gradient method (related [post]({% post_url 2020-12-21-policy-gradient %})), the gradient that we used to update the policy parameters is

$$
\triangledown_\theta J(\theta) = \frac{1}{N} \sum_i \bigl( \sum_t \triangledown_\theta \log \pi_{\theta} (a_{i,t} \vert s_{i,t}) \bigr) \bigl( \sum_{t'=t}^T r(s_{i,t'}, a_{i,t'}) -  \frac{1}{N}\sum_i^N \sum_{t'=t}^T r(s_t', a_t') \bigr)
$$

Note that some parts of the above are equivalent to the value functions we just mentioned:

$$
Q^\pi(s_t, a_t) = \sum_{t'=t}^T r(s_{i,t'}, a_{i,t'}) \\
V^\pi(s_t) = \frac{1}{N}\sum_i^N \sum_{t'=t}^T r(s_{i,t'}, a_{i,t'})
$$

The RL objective cam be rewritten as

$$
\triangledown_\theta J(\theta) = \frac{1}{N} \sum_i \bigl( \sum_t \triangledown_\theta \log \pi_{\theta} (a_{i,t} \vert s_{i,t}) \bigr) \bigl( Q^\pi(s_t,a_t) - V^\pi(s_t) \bigr)
$$

We call $$A^\pi(s_t, a_t) = Q^\pi(s_t,a_t) - V^\pi(s_t)$$ **advantage**, a measure of how good $$a_t$$ is in terms of maximizing rewards relative to the average in that state.

Recall Q-function is the total rewards from taking action $$a_t$$ in state $$s_t$$. We directly know the reward at $$t$$; we just call $$r(a_t, s_t)$$, but from $$t+1$$, the expected reward will be approximately $$V^\pi(s_{t+1})$$. Therefore, advantage function can be rewritten as

$$
A^\pi(a_t,s_t) = r(a_t, s_t) + V^\pi(s_{t+1}) - V^\pi(s_t)
$$

Now there's only one value function in our RL objective. That's great news, because we can fit a model to predict $$V^\pi$$ that would give a smoother gradient to $$J(\theta)$$. Using the Monte-Carlo method (fancy way to saying just sample a lot and take the average), our target is approximated as $$\frac{1}{N} \sum_i^N \sum_t^T r(a_{i,t}, s_{i,t})$$. We can sample the policy to get pairs of $$(s_t, y_{i,t})$$, where $$y_{i,t} = \sum_t^T r(a_t, s_t)$$. Since the value function is equivalent to the expected value of q-values, we can use use this $$y_{i,t}$$ directly and do a regression that minimizes $$\sum_i \left\lVert \hat{V}(s_i) - \sum_{t'=t}^T r\right\rVert^2 $$.

Now we will switch between fitting on this value function ($$V^\pi$$), a.k.a., critic and improving the policy $$\pi$$, a.k.a., actor. The method name should make sense now because value function gives you the expected cumulative rewards given a state, thereby "criticizing" your policy, while the policy is one that determines what actions to take, thereby taking "actions".

During the training process, we switch back and forth between improving the critic by better estimating the expected total rewards at any given state AND improving the actor with the help of the better critic. The step-by-step process is

1. generate sample pairs $$(s_i, a_i)$$ from $$\pi_\theta$$ and record the rewards
2. fit the value function $$\hat{V}_\phi^\pi(s)$$ to the sum of rewards to go (q-value)
3. estimate advantage $$\hat{A}^\pi(s_i, a_i) = r(a_t, s_t) + V^\pi(s_{t+1}) - V^\pi(s_t)$$
4. calculate the gradient of the RL objective $$\triangledown_\theta J(\theta) = \sum_i \triangledown_\theta \log \pi_\theta(a_i\vert s_i) \hat{A}^\pi(s_i, a_i)$$
5. Update the parameters of the model $$\theta \leftarrow \theta + \alpha \triangledown_\theta J(\theta)$$
6. Repeat 1-6.

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

For some detailed pytorch-specific explanation of the code, refer to the [post]({% post_url 2020-12-20-reinforcement-learning-primer-rewards %}). `sample` function returns the results of the rollouts of your policy.

#### Discount factor
One common thing to see in estimating $$V^\pi(s)$$ is to discount the future. The usefulness of this becomes more apparent when your time horizon infinite.

$$V^\pi(s) = E_{a \sim \pi} [\sum_{t=0}^\infinity r(s_t, a_t \vert s_0 = s)] $$

In the above case, we will never finish finding out $$V^\pi$$. Introducing a discount rate $$\gamma$, the value function now becomes

$$V^\pi(s) = E_{a \sim \pi} [\sum_{t=0}^\infinity \gamma^t r(s_t, a_t \vert s_0 = s)] $$

and now our value function estimation will use the above as the target (step 2), and the advantage will also discount the next observation by $$\gamma$$, i.e., $$\hat{A}^\pi(s_i, a_i) = r(a_t, s_t) + \gamma V^\pi(s_{t+1}) - V^\pi(s_t)$$

#### Pros and Cons
Actor-critic method provides lower variance than policy gradient method and tends to converge well. It's more **sample-efficient** in that you can converge to a model with fewer samples than a policy gradient method could be. But note that our actor is always sampling based on its own policy; this means that it may not take actions that could be potentially very good but are drastically different from what it has tried. (Such a method is called an on-policy algorithm, meaning it samples according to its policy.) That means your actor can be great at converging into the local maximum, but can be stuck and **not finding the global maximum**.

This leads well into Q-learning, which is an **off-policy** algorithm, another great method in reinforcement learning.

