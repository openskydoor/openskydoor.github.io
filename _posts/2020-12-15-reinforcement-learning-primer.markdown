---
layout: post
title: "Reinforcement Learning Primer"
categories: reinforcement_learning
---
{% include math.html %}

Reinforcement learning (RL) is an area of machine learning. A most widely used and understood area of machine learning is supervised learning, so I think it's easy to understand what it is by comparing it against supervised learning. In supervised learning, you know inputs and labels (the truth), and your goal is to come up with a model that can guess the labels only based on inputs. One example will be given some facts about a home, you would like to guess what the market price of the home is. You guess will be your prediction, and the truth value is what it gets sold in real life.

In reinforcement learning, you do not have this truth values, only whether your "inputs" (policy, to be better defined later) are good or bad and to what degree they are good or bad.

When would this kind of problem-framing be useful? Virtually all the problems that have different ways to achieve a certain outcome. For example, when you play a board game like Catan and you win, you don't win with the same sequence of actions all the time. Your board setup is different every time, you take different actions every time, but you have a consistent scoring mechanism that advises you on which actions to take.

There are a few ways that reinforcement learning is different from supervised learning:

1. your inputs are not independent and identically distributed (i.i.d.). In the Catan example, your game state changes depend on your action, (which affects your opponents' actions, etc)!
2. ground truth is not known. Going back to the Catan example, you have different ways to win even given the same initial board setup, and it would be very expensive to come up with all different possible ways to win and all possible actions your opponents can take!

Some other examples that reinforcement learning is used in solving are physical tasks, such as moving an item from one place to another,games like Atari and Go, and autonomous driving.

How should we frame these kinds of problems so that we can attempt at solving?

A markov decision process (MDP) is a great way to formalize the problem. In a Markov decision process, you deal with four variables

S: state

A: action

P: probability of action taken given state

R: immediate reward per action

//TODO:insert markov diagram

Sometimes you don't know the real state, however. For example, when you are playing a poker, you don't have full information; the only thing you can do is to observe. You can use your observation to infer the true state, and your decision process will look more like partially observed Markov decision process

//TODO:insert with observation diagram

In reinforcement learning, our goal is to learn what will happen to the rewards when I take certain actions in a given state. We call this a **policy**, typically represented with $$\pi$$.

### Imitation Learning

So how might we learn a policy? I mentioned earlier that reinforcement learning is different from supervised learning in that your inputs are not i.i.d., and your ground truth is not known, i.e., there isn't necessarily the one and only correct policy. But what if there's a good example to emulate, so there's a well agreed-upon ground "truth"? For example, if we wanted a self-driving car, can we just feed a bunch of humans' actual driving routes (if an self-driving car would drive like I do, it might be good enough for me... I'm still alive!) and have the car learn to replicate? Well, that sounds a lot like supervised learning, and it is! We call this **imitation learning/behavior cloning**. So let's talk about it before we discuss other fancy RL algorithms.

The steps to be taken are:

1. collection demonstrations from an expert, e.g., me driving in the above example. This will consist of multiple independent sequences of action, or **trajectories** $$ \tau $$.

2. treat demonstrations as i.i.d. state-action pairs $$(s_{0}, a_0), (s_1, a_1), ...$$.

3. train a policy $$\pi_\theta$$ that minimizes a loss function $$L(\pi_\theta(s), a)$$.

What might go wrong with this approach?

The biggest problem is the distributional shift between the expert demonstrations and policy. Suppose I demonstrated a right turn, but the model, for various reasons to be explained later, wanted to go straight, the car ends up in a state that it's never seen before during the training phase, because I never demonstrated what I would do in that strange road! The policy, which was never trained with a particular state, takes an action that is suboptimal, and it continues to move further and further from the training trajectories.

A solution to this is to collect more data in that state. In the self-driving car example, a human may annotate what action to take in the new state that the policy ends up in, augment the training data, and train the policy again. We call this practice **DAgger (Dataset Aggregation)**.

There are still other problems, such as humans not being consistent in their demonstration (Non-Markovian behavior), multimodal behavior (multiple acceptable trajectories demonstrated given the same state). You can try mitigating these problems using a Gaussian mixture model, introducing latent variables, etc.

For these reasons, the supervised learning approach to reinforcement learning tends to underperform other RL algorithms (we'll discuss soon!). Rewards-based RL algorithms tend to work better than imitation learning, but it is useful when the right reward function is ambiguous, rewards are not frequent, or when you want with a reasonable baseline to start training your rewards-based RL algorithm. A more in-depth discussion of rewards-based RL algorithms will start from the next post.
