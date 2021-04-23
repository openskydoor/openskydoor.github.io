---
layout: post
title: "Intro to Double Deep Q-learning"
categories: reinforcement_learning
redirect_from:
    - /post/q-learning/
---
{% include math.html %}

Q-learning is a type of value-based methods in reinforcement learning (RL) in which you don't learn a policy explicitly; instead you just learn the Q-function. In this world, you would simply take an action that maximizes the Q-function. In this post, I'm going to focus on double deep q-learning (DQN), a flavor of Q-learning that has shown superior performance to classic DQN.

#### Policy
In Q-learning, we're foregoing learning the policy entirely and keeping the policy as choosing any action that returns the maximum value from Q-function. That means our policy $$\pi$$ is essentially

$$
\pi(a_t|s_t) = \begin{cases}
1 \text{ if } a_t = \mathop{\operatorname{arg\,max_{a_t}}} Q_\phi (s_t, a_t) \\
0 \text{ otherwise}
\end{cases}
$$

Compared to the actor-critic method, we are skipping the step 4 and 5.

1. generate sample pairs $$(s_i, a_i)$$ from $$\pi_\theta$$ and record the rewards
2. fit the value function $$\hat{V}_\phi^\pi(s)$$ to sampled reward sums (q-values)
3. estimate advantage $$\hat{A}^\pi(s_i, a_i) = r(a_t, s_t) + \gamma V^\pi(s_{t+1}) - V^\pi(s_t)$$
4. <s>calculate the gradient of the RL objective</s>
5. <s>Update the parameters of the model</s>

In the actor-critic method, we use $$A^\pi$$ to nudge the policy to move in the right direction. We are still doing the same in Q-learning, but note due to our policy's deterministic nature, $$V^\pi(s) = \max_a Q^\pi(s, a)$$. This means we can just focus on learning Q-function and still end up with the same intention. It should now make sense why this method is called "Q-learning".

#### Bellman equation
Bellman equation is a recursive formula for q-function.

Recall the identity, if using the discount factor $$\gamma$$:

$$
Q^\pi(s_t, a_t) = r(s_t, a_t) + \gamma V^\pi(s_{t+1})
$$

Since $$V^\pi(s_t) = E[Q^\pi (s_t, a_t)]$$,

$$Q^\pi(s_t, a_t) = r(s_t, a_t) + \gamma E[Q^\pi (s_{t+1}, a_{t+1})]$$

Since our policy is one that chooses an action that maximizes q-function, we approximate $$E[Q(s_{t+1})] \approx \max_{\boldsymbol{a_{t+1}}} Q_\phi (s_{t+1}, a_{t+1})$$. We can define our target $$y$$ for fitting $$\hat{Q}^\pi$$ as

$$y = r(s_t, a_t) + \gamma \max_a Q^\pi (s_{t+1}, a_{t+1})$$

Our goal is to train $$Q_\phi$$ such that $$ r(s_t, a_t) + \gamma \max_t Q(s_{t+1}, a) - y$$ is minimal. This error is called Bellman error.

What I have described so far is relevant to any Q-learning. The next two items I discuss are specific to double DQN.

#### Target Network
Let's take a closer look at our $$Q_\phi$$ gradient.

$$
\phi \leftarrow \phi - \alpha \frac{dQ_\phi}{d\phi} (s_i,a_i)(Q_\phi(s_i,a_i) - y_i) \\
= \phi - \alpha \frac{dQ_\phi}{d\phi} (s_i,a_i)(Q_\phi(s_i,a_i) - [r(s_t, a_t) + \gamma \max_\boldsymbol{a} Q_\phi (s_{t+1}, a_{t+1})])
$$

Our target value $$y_i$$ includes the term that we are differentiating against. This slows down learning because the target value $$y_i$$ being correlated with what we are trying to improve $$Q_\phi$$ does not help the model converge very well! As a symptom of this correlation, $$Q_\phi$$ can sometimes overestimate the q-values of certain actions.

We can solve this issue by using another q-function $$Q_{\phi'}$$ in the target value. We call this $$Q_{\phi'}$$ the target network. So instead of using the same network to both calculate the target value and optimize, we use two different ones. Once in a while, we will swap these two networks, so that $$Q_\phi$$ becomes $$Q_{\phi'}$$, and vice versa.

Yet another trick to reduce correlation between $$Q_\phi$$ and $$y_i$$ is using the action per $$Q_\phi$$ in calculating $$y_i$$ instead of $$Q_{\phi'}$$, even though the value itself is still calculated with $$Q_{\phi'}$$, i.e.,

$$y_t = r(s_t, a_t) + \gamma \max_{\boldsymbol{a_{t+1}}} Q_{\phi'} (s_t, \mathop{\operatorname{arg\,max_{\boldsymbol{a_{t+1}}}}}Q_\phi(s_{t+1}, a_{t+1}))$$


#### Off-policy Sampling

Q-function algorithm is an off-policy algorithm; this means that we don't use the algorithm's policy to generate data that we train with. One effective way to implement this is to use a replay buffer. You generate data by sampling with some policy and put them in the replay buffer $$B$$. During the training, you sample a batch from the buffer uniformly. This will allow the samples to be not correlated. You also update the $$Q_\phi$$ with a batch of data, allowing the gradient variance to stay low.

#### Training Steps

Now we're fully ready to put together what we've learned so far and outline the training steps.

1. update target network parameters $$\phi' \leftarrow \phi$$
     - Repeat $$N$$ times
    2. collect samples, add to the replay buffer $$B$$
         - Repeat $$K$$ times
        3. sample a batch of $$(s, a, s', r)$$ from $$B$$
        4. update the parameter $$\phi \leftarrow \phi - \alpha \frac{dQ_\phi}{d\phi} (s_t,a_t)(Q_\phi(s_t,a_t) - [r(s_t, a_t) + \gamma \max_{\boldsymbol{a_{t+1}}} Q_{\phi'} (s_t, \mathop{\operatorname{arg\,max_{\boldsymbol{a_{t+1}}}}}Q_\phi(s_{t+1}, a_{t+1}))])$$

Here's the pseudo-code:
```python
from typing import Tuple
import torch
from torch import nn
from torch import optim
import numpy as np

class ArgMaxPolicy:

    def __init__(self, critic: nn.Module) -> None:
        self.critic = critic

    def get_action(self, observation: np.array) -> np.array:
        q_values = self.critic.qa_values(observation)
        actions = np.argmax(q_values, axis=1)
        return actions.squeeze()

class DoubleDQNCritic:

    def __init__(self,
                observation_dim: int,
                action_dim: int,
                hidden_layer_dim: int,
                learning_rate: float,
                gamma: float) -> None:
        self.q_net = nn.Sequential(nn.Linear(observation_dim, hidden_layer_dim), nn.Linear(hidden_layer_dim, action_dim))
        self.q_net_target = nn.Sequential(nn.Linear(observation_dim, hidden_layer_dim), nn.Linear(hidden_layer_dim, action_dim))
        self.optimizer = optim.Adam(self.q_net.parameters(), learning_rate)
        self.loss_func = nn.MSELoss()
        self.gamma = gamma

    def update(self,
                observations: torch.Tensor,
                actions: torch.Tensor,
                next_observations: torch.Tensor,
                rewards: torch.Tensor) -> None:
        # get q values for all actions
        qa_t_values = self.q_net(observations)
        # get q values for actions actually performed
        q_t_values = torch.gather(qa_t_values, 1, actions.unsqueeze(1)).squeeze(1)

        # get target q values for all actions
        qa_tp1_target_values = self.q_net_target(next_observations)v].detach()
        # get target q values for actions that policy per q_net would have taken
        q_tp1_target_values = (torch.gather(
                                qa_tp1_target_values,
                                1,
                                torch.argmax(q_t_values, dim=1).unsqueeze(1)
                                ).squeeze(1)

        target = rewards + self.gamma * q_tp1_target_values
        self.optimizer.zero_grad()
        loss = self.loss_func(q_t_values, target)
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        for target_param, param in zip(self.q_net_target.parameters(),
                                         self.q_net.parameters()):
            target_param.data.copy_(param.data)

def collect_samples() -> None:
    """Generate samples based on some random policy and add to buffer."""
    ...

def sample_from_buffer() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample from replay buffer."""
    ...

def batch_train(observation_dim: int,
                action_dim: int,
                hidden_layer_dim: int,
                learning_rate: float,
                gamma: float):
    critic = DoubleDQNCritic(observation_dim, action_dim, hidden_layer_dim, learning_rate, gamma)

    for _ in range(NUM_BATCHES_PER_ITER):
        collect_samples(actor)
        observations, actions, rewards, next_observations = collect_samples()
        for _ in range(NUM_UPDATES_PER_BATCH):
            critic.update(observations, actions, next_observations, rewards)

    critic.update_target_network()
```

#### Pros and Cons
Q-learning is beneficial since it has lower variance updates; you don't need as many samples to improve your model as, e.g., a policy gradient method would require.

One drawback is q-learning as I've described above will not work well with problems with continuous action. Note the $$max$$ function in the target value $$r(s_t, a_t) + \gamma \max_{\boldsymbol{a_{t+1}}} Q_{\phi'}$$. Trying to find the max value in some continuous space will require infinite samples! You could ignore and just take the maximum of what you've sampled so far, but that will not be very accurate. There are a few options here to allow you to still use q-learning, such as with better optimization or learning an approximation of this problematic function $$max$$, but further details will be a topic for another post.


