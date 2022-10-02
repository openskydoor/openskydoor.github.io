---
layout: post
title: "Engineer Ruminates on Management (2) - On Hierarchy"
categories: others
---

## Remove Hierarchy of Opportunities

Companies usually have a level guideline for engineers. As you go higher in the chain, you are evaluated more on "leadership"--whatever that means--than your implementation. Essentially, the kinds of skills you are expected to have are different when you become a "senior" personnel. (I'm not the only one to note this. Andrew Ng in deeplearning.ai's newsletter once pointed out a senior engineer is a very different position from non-senior engineer positions!)

### Why?

Regardless of your skill set, you don't get the opportunity to "lead" until you have been a mid-level engineer for some time, because you need to go through this arbitrary pipeline of opportunities. This is so puzzling to me. I have seen many new grads that were better tech communicators than me or just as good "coders" as myself; I've also seen senior ICs that were terrible coders or did not understand the basic principles of systems design. But it didn't matter what skill sets these people had; the "senior" engineer always wrote the spec and held meetings, while the new grads executed the tasks. Why aren't they allowed to collaborate by using their better skills? Are people just rent-seeking and too timid to admit that these new grads can be more productive than themselves?

### The Pipeline

I'm not advocating for zero hierarchy. (I simply don't have an opinion on whether that is a good thing.) What I do find problematic is the hierarchy of opportunities. It is a form of hazing in engineering organizations. As a new grad, you write code, and if you do that for a year or two without being a terribly annoying person to work with, you get to the next level. At this stage, you may not even be invited to product or engineering design review meetings. This next level means you can handle tasks without much supervision for the most part. You're most likely to be doing the same thing you have already been doing: code. But the next level expects you to do something you have not been asked to do: demonstrate leadership. What that in reality translates to is to be someone your manager likes talking to; more concretely, you will need skills like understanding the progress of the project and the relationship with the product goals, communicating it in manners that are easy for the manager to understand (they like organization and abstractions), and proactively resolving conflicts with any stakeholders. When your manager for some reason thinks you're ready to prepare for the next level promotion, you will be asked to write design docs and lead some meetings, because these are the concrete materials that your manager can use to make a case for your promotion. For the next next level, it will be similar things, but the scope of your project will be bigger, and the impact of your feature/product matters a lot too, even when the product decision is made by your product manager... so now you get political and try to land on a more impactful project.

One might argue you need the coding skills before you can do a system design and manage projects, but that implies that when you interview, if you can ace the system design portion, you will definitely do well in coding, which is clearly not true... Knowing how to code "well" helps, but this is a very inefficient way to learn to design systems. Perhaps I am missing something here, and there is an inherent boost to one's leadership and design skills when one has been coding for a few years, and this pipeline is all well justified. Maybe this hazing is a way for managers to de-risk their project, because there probably is some correlation between the participating engineers' years of experience and project success. And I'm just a weirdo to question this.

### Hella Senior Engineer Problem

I think different skills need to be nurtured earlier and _also later_. Have you ever been frustrated by some decision made by higher-ups and felt they don't understand the real important problems? They tend to arise because they are not expected to understand what lower level ICs do at depth. I have seen a head of analytics who didn't know SQL (I think because she did all of her work in Excel as an IC) and wasn't sure whether learning this was worth her time. It might have been justified because she's more of a people manager at that point, but you could imagine how she would not be best equipped to make certain decisions. What the upper level continues to do to mitigate this problem is to send out surveys to learn how ICs feel/think/want, but that does not give them insight. Quantitative understanding is important and can be delegated, but qualitative understanding is actually an efficient way to approach uncertainty and vague problems and cannot be delegated, i.e., they actually have to spend time to understand what ICs are doing at a deeper level.

Note that I'm not advocating for removing hierarchy because I want to promote fairness. As I mentioned earlier, my primary motivation is to build a productive team. Assuming in the short-term the individual productivity is constant, this translates to making the team more efficient. Right now I see so much of responsibilities are doled out based on this arbitrary pipeline of career progression and not based on strength, and there is a lack of aggressive and selective nurturing. The pipeline is an easy thing to default to, but I would like to work with a manager who identifies weaknesses and strengths in ICs and strategically uses them for different projects regardless of their titles.

In this series:

- [Write More Effectively]({% post_url 2021-08-13-managing-writing%})

- [Remove Hierarchy of Opportunities]({% post_url 2021-08-14-hierarchy%})

- [ICs Must Study]({% post_url 2021-08-15-studying%})

- [Measure Concrete Actions of ICs, Micromanage ICs' Career]({% post_url 2021-08-16-measuring%})
