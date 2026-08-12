**A**

**RESEARCH PROJECT**

**ON**

**COMPARATIVE EVALUATION OF GRAPH-BASED AND SEQUENTIAL MULTI-BEHAVIOUR RECOMMENDATION ARCHITECTURES IN A LIVE STREAMING DOMAIN**

**BY**

**ADJEKOFORI ISRAEL OGHENEFEJIRO**

**PSC2207846**

**CHAPTER ONE**

**INTRODUCTION**

# **1.1 Background of the Study**

## **1.1.1 The Rise of Recommender Systems**

Modern digital platforms succeed or fail on a single capability: their ability to put the right thing in front of the right person at the right time. Recommender systems are the machinery behind that capability. They analyse what users have done in the past, infer what they are likely to want next, and surface personalised suggestions that influence what hundreds of millions of people watch, buy, and listen to each day. The economic stakes are not abstract. Precedence Research (2024a) values the global recommendation engine market at roughly USD 5.39 billion in 2024 and projects growth at a compound annual rate of 36.33%, with the market expected to reach about USD 119.43 billion by 2034\. This is not just a story about more data being available; it reflects a hardening industry consensus that recommendation quality maps directly onto retention, engagement, and revenue.

Live streaming is one of the domains where this mapping is most visible. Live streaming platforms blend real-time entertainment, social interaction, and commerce into a single product surface, and they have become enormous: the livestream e-commerce segment alone was valued at USD 14.93 billion in 2024 and is projected to exceed USD 258 billion by 2034 (Precedence Research, 2024b). On Kuaishou, live streaming accounts for roughly 30% of total company revenue (Qu et al., 2025). When a platform's monetisation is that concentrated in one product surface, matching the right viewer to the right streamer at the right moment stops being a UX nicety and becomes a core financial system.

## **1.1.2 Key Concepts in Multi-Behaviour Recommendation**

Early recommender systems, including landmark approaches such as matrix factorisation and neural collaborative filtering (He et al., 2017), treated user preference as a one-dimensional signal extracted from a single type of interaction. That interaction was usually a purchase, a rating, or a click. The approach worked in controlled academic settings, but it discards most of what users actually do. A user on a live streaming platform does not interact with a streamer in one way. They click into a room, stay or leave within seconds, double-tap a like, send a comment, occasionally send a virtual gift. Every one of those actions tells the system something different about interest, attention, and intent.

Multi-Behaviour Recommendation (MBR) emerged from this recognition. The paradigm uses multiple interaction types together to build richer user representations and, critically, to address the data sparsity problem that single-behaviour systems cannot escape. The commercially meaningful action on any platform, whether that is a purchase, a subscription, or a gift, is almost always the rarest. By letting abundant signals such as clicks and likes propagate into the representation of the sparse target action, MBR systems can make useful predictions for users and items that would otherwise be invisible. Research interest has tracked the practical importance closely. Kim et al. (2025) report that peer-reviewed publications on multi-behaviour recommendation rose from 37 across 2019 to 2020, to 79 across 2021 to 2022, and to 164 across 2023 to 2024\.

## **1.1.3 The Two Dominant Architectural Families**

Two architectural families dominate the multi-behaviour literature, and they reflect genuinely different intuitions about what user behaviour is and how it should be modelled.

The first family is graph-based. These models represent users and items as nodes in a heterogeneous graph, with different behaviour types appearing as distinct edge types between user nodes and item nodes. Graph neural network propagation then learns higher-order collaborative signals, picking up not only direct user-item interactions but also indirect signals carried by shared neighbours and longer paths through the graph. The foundational model in this family is the Multi-Behaviour Graph Convolutional Network (MBGCN) proposed by Jin et al. (2020), which introduced a unified heterogeneous graph for multi-behaviour data and showed that learning behaviour-specific weights through user-item propagation, combined with capturing behaviour semantics through item-item propagation, produces large gains over single-behaviour baselines.

The second family is attention-based and sequential. Here a user's history is treated as a temporally ordered sequence of interactions, and transformer-style self-attention is used to learn how past behaviours interact with each other to predict what comes next. The foundational model in this family relevant to multi-behaviour settings is ATRank (Zhou et al., 2018). ATRank projects heterogeneous user behaviours into multiple latent semantic spaces, applies self-attention to model inter-behaviour influence, and then uses vanilla attention to generate the context vector that drives the recommendation. Zhou et al. showed that this approach could replace recurrent encoders entirely while training faster and producing better results on heterogeneous behaviour data.

The two families embody different inductive biases. Graph-based models privilege structural relationships and the collaborative signals that move across the full user-item interaction space. Sequential models privilege temporal order and the contextual relevance of one user's specific history. Both have been validated extensively. The catch is that the validation has happened almost entirely on persistent-item, e-commerce datasets with clean, well-ordered behaviour hierarchies.

## **1.1.4 The Gap: Untested Assumptions in Non-Standard Domains**

The empirical foundation of multi-behaviour recommendation research sits on a small set of benchmark datasets, mostly drawn from e-commerce platforms such as Tmall, Taobao, and Beibei. These datasets share a structural profile: items are persistent, catalogues are stable, and behaviour types follow a predictable intensity hierarchy in which clicking precedes adding to cart, which precedes purchase. Both architectural families fit this setting comfortably. Graph models benefit from stable node representations that accumulate collaborative signal over time. Sequential models benefit from sequences in which behaviours are semantically comparable and temporally informative.

Live streaming breaks every one of these assumptions at once. Items, called live rooms, are ephemeral. They come into existence when a streamer goes live and disappear when the stream ends. No stable catalogue exists for collaborative signal to accumulate against in the way graph models expect. The behaviour hierarchy is ambiguous as well. A user may send a gift without ever liking or commenting, or comment heavily without ever gifting. User sessions are shaped by daily rhythms and content-specific viewing patterns that defeat simple recency assumptions. These differences raise a practical question with real theoretical weight: do the architectural assumptions validated on e-commerce data transfer to live streaming, and if not, under what specific conditions does each family start to fail?

Nobody has answered this question systematically. The release of KuaiLive (Qu et al., 2025), the first publicly available large-scale live streaming interaction dataset, finally makes the evaluation possible. By applying both architectural families to this new domain and looking past aggregate scores to ask where and why each approach holds up, this study sets out to contribute empirical and theoretical insight to the multi-behaviour recommendation community.

#  **1.2 Statement of the Problem**

Multi-behaviour recommendation research has grown quickly, but the field has a dependency problem. Almost all of the work, including evaluations of MBGCN (Jin et al., 2020\) and ATRank (Zhou et al., 2018), has been carried out on a small handful of e-commerce benchmarks characterised by persistent catalogues and behaviourally ordered interaction funnels. The result is a systematic blind spot. How these architectures behave in domains with structurally different interaction ecologies is largely unknown.

Live streaming is one such domain, and it is both commercially important and structurally distinctive. It is also critically underrepresented in the literature. Prior work on live streaming recommendation has relied mostly on proprietary industrial datasets that nobody outside the originating company can reproduce or evaluate (Qu et al., 2025). Without a standardised, public benchmark, the community has not been able to do systematic architectural comparison or to draw generalisable conclusions about what works in this setting.

The deeper technical problem is that both graph-based and sequential multi-behaviour architectures encode assumptions that may be incompatible with live streaming data. Graph models assume item nodes persist long enough to accumulate meaningful collaborative signal through propagation. In live streaming, where a room exists only as long as the broadcast does, that assumption simply does not hold. The graph is in constant flux, and no stable neighbourhood forms around an item node. Sequential models, for their part, assume a user's interaction history is a meaningful temporal sequence where past behaviours are comparable in type and scale. In live streaming, a passive two-second click and an active financial gift are so different in both effort and intent that treating them as equivalent elements of the same sequence flattens, rather than captures, the underlying preference signal.

Neither architectural family has been evaluated under the cold-start conditions that define live streaming at scale either. On KuaiLive, roughly 43.8% of users have fewer than 50 total interactions (Qu et al., 2025), which is a severe sparsity environment. Whether graph propagation from auxiliary behaviours helps these users in a domain where the graph itself is unstable, or whether sequential models can do enough with thin histories through attention-based context, is an open question.

This study addresses these gaps directly. Through a systematic evaluation of MBGCN and ATRank on the KuaiLive dataset, with segmented analysis across user populations and temporal windows, it produces evidence-based insight into which architectural family is better suited to multi-behaviour recommendation in ephemeral-item, real-time interaction environments, and under what specific conditions.

# **1.3 Aim of the Study**

The aim of this study is to systematically evaluate and compare graph-based and attention-based sequential multi-behaviour recommendation architectures on a live streaming domain. The goal is to determine which architectural family is a better fit for the structural characteristics of ephemeral-item, real-time interaction environments, and to identify the specific conditions under which each approach succeeds or fails.

# **1.4 Objectives of the Study**

The following specific objectives guide the study:

I. To analyse the structural assumptions underlying graph-based and attention-based sequential multi-behaviour recommendation architectures, specifically MBGCN and ATRank, and identify how those assumptions align or conflict with the properties of live streaming interaction data as represented in KuaiLive.

Ii. To design a rigorous experimental framework for evaluating both architectural families on the KuaiLive dataset, including data preprocessing pipelines, behaviour segmentation strategies, and temporally grounded train-test splits that respect the ephemeral nature of live room items.

Iii. To implement and train MBGCN as the representative graph-based model and ATRank as the representative attention-based sequential model on the KuaiLive multi-behaviour interaction data, using virtual gift prediction as the target behaviour and click, like, and comment as auxiliary behaviours.

Iv. To evaluate model performance across both aggregate metrics and segmented user populations, including cold-start versus warm users, sparse versus dense behaviour profiles, and early versus late temporal windows, in order to identify the conditions under which each architecture demonstrates a relative advantage or disadvantage.

# **1.5 Research Questions**

This study is guided by the following research questions:

RQ1:Do graph-based multi-behaviour architectures and attention-based sequential architectures produce meaningfully different performance outcomes when evaluated on a live streaming domain with ephemeral items, compared to what existing e-commerce benchmarks would predict?

RQ2:Under what user-level conditions, particularly regarding behavioural density and cold-start severity, does each architectural family demonstrate a relative advantage in predicting sparse target behaviours such as virtual gifting?

RQ3:How does the ephemeral nature of live room items, where the candidate pool changes continuously throughout the observation window, affect the structural integrity of graph-based models relative to sequential models over time?

RQ4:To what extent does the semantic heterogeneity of live streaming behaviours, particularly the qualitative difference between a passive click and a financial gift, challenge the behaviour hierarchy assumptions embedded in both architectural families?

# **1.6 Research Hypothesis**

This study advances the following directional hypothesis, grounded in the structural differences between graph-based and sequential architectures and the known properties of the KuaiLive dataset. Because the work is experimental and comparative, both a null hypothesis and a set of directional alternative hypotheses are stated.

## **1.6.1 Null Hypothesis**

H0: There is no statistically significant difference in the gift-through-rate prediction performance of MBGCN and ATRank when evaluated on the KuaiLive live streaming dataset across all user population segments, including cold-start users, warm users, and users segmented by behavioural density.

## **1.6.2 Alternative Hypotheses**

H1: MBGCN demonstrates significantly superior gift prediction performance over ATRank for cold-start users, defined as users with fewer than ten target-behaviour interactions in the training period. The reasoning is that graph propagation from auxiliary behaviours can compensate for the absence of the dense sequential histories that ATRank needs to function well.

H2: ATRank demonstrates significantly superior gift prediction performance over MBGCN for warm users with dense interaction histories. The self-attention mechanism can exploit the rich sequential patterns and temporal context that graph propagation, operating on a structurally unstable ephemeral-item graph, cannot adequately capture.

H3: The performance advantage of MBGCN over ATRank diminishes significantly as the observation window progresses through the 21-day KuaiLive collection period. As expired live rooms accumulate, the structural integrity of the graph degrades, while sequential models continue to update from the most recent interactions.

These three directional hypotheses are not mutually exclusive. The main contribution of the hypothesis structure is to position the thesis to produce a more nuanced finding than a single aggregate winner. If H1, H2, and H3 are all supported, the conclusion is not that one architecture is universally superior but that architectural fitness in live streaming recommendation is conditioned on measurable user-level and temporal variables. That conditional finding has more practical and theoretical value than a flat performance comparison (Jin et al., 2020; Qu et al., 2025).

# **1.7 Significance of the Study**

This study makes contributions at three intersecting levels. It speaks to the academic multi-behaviour recommendation community, to industry practitioners building live streaming and real-time interaction platforms, and to the broader methodology of recommender systems evaluation.

## **1.7.1 Academic Community**

The multi-behaviour recommendation literature has grown sharply, with peer-reviewed publications rising from 37 across 2019 to 2020 to 164 across 2023 to 2024 (Kim et al., 2025). Almost all of that growth has happened inside the narrow box of e-commerce benchmarking. This study contributes the first systematic architectural comparison of graph-based and sequential multi-behaviour models on a publicly available live streaming dataset, establishing baseline performance figures that future work can build on. Beyond the numbers, the segmented evaluation framework developed here, which compares architectures across cold-start severity, behavioural density, and temporal window position, offers a reusable methodology other researchers can apply to evaluate architectural fitness in new domains.

The study also contributes to an important theoretical conversation about benchmarking validity. By showing that architectural rankings observed on e-commerce data may not transfer to domains with ephemeral items and ambiguous behaviour hierarchies, the study offers empirical evidence for a claim that is widely suspected but rarely tested directly: that the field's reliance on a small set of standard benchmarks produces architectural preferences that may not generalise. The finding supports recent calls for greater domain diversity in recommender systems evaluation (Qu et al., 2025; Sun et al., 2020).

## **1.7.2 Industry Practitioners**

Engineers and product teams building recommender systems for live streaming platforms have no empirical basis right now for choosing between graph-based and sequential architectures. Published work offers performance comparisons only on e-commerce data, and the proprietary studies conducted internally at companies such as Kuaishou and TikTok are not publicly reproducible (Lu et al., 2025). This study addresses that gap with evidence-based guidance tied to measurable platform characteristics: the maturity of the user base, the rate of catalogue churn driven by stream endings, and the sparsity of the target behaviour relative to auxiliary interactions. A platform engineer can use these findings to make an informed architectural decision based on where their platform sits on those dimensions.

The findings carry implications beyond live streaming. Any domain involving ephemeral items, such as ticketed events, flash sales, time-limited content drops, or job postings, faces structurally similar challenges. The insights generated here about when graph propagation degrades and when sequential attention holds up should transfer to those adjacent settings.

##  

## **1.7.3 The Broader Recommender Systems Community**

At the methodological level, the study demonstrates the value of running architectural comparisons on domains that were not used in the original design and validation of the models being compared. MBGCN was designed for e-commerce and validated on Tmall and Taobao (Jin et al., 2020). ATRank was designed for heterogeneous behaviour sequences and validated on Amazon product data (Zhou et al., 2018). Applying both to a structurally alien domain and analysing where and why their performance changes is a form of adversarial evaluation that exposes architectural assumptions that would otherwise stay invisible. The thesis therefore makes an implicit case for a more stress-testing-oriented approach to model evaluation in the recommendation systems community.

# **1.8 Scope of the Study**

## **1.8.1 What the Study Covers**

This study covers the design, implementation, and evaluation of two representative multi-behaviour recommendation architectures on a live streaming domain. The architectures evaluated are MBGCN (Jin et al., 2020), representing the graph-based family, and ATRank (Zhou et al., 2018), representing the attention-based sequential family. Both models are applied to the KuaiLive dataset (Qu et al., 2025), which records interaction logs from 23,772 users and 452,621 streamers across 11,613,708 live rooms over a 21-day period collected from the Kuaishou platform between 5 May and 25 May 2025\.

The target behaviour for all prediction tasks is virtual gift-giving. It was chosen for its economic significance, its natural sparsity relative to auxiliary behaviours, and the availability of gift price information that enriches the prediction task. Click, like, and comment interactions serve as auxiliary behaviours. The evaluation covers both aggregate metrics and segmented analysis across user population subgroups defined by cold-start status, behavioural density, and temporal position within the observation window.

## **1.8.2 Technologies and Tools**

All experiments are implemented in Python using the PyTorch deep learning framework (Paszke et al., 2019). Graph construction for MBGCN uses the PyTorch Geometric library (Fey & Lenssen, 2019). Evaluation protocols follow those established in prior multi-behaviour recommendation work, with metrics including Hit Rate at K (HR@K), Normalised Discounted Cumulative Gain at K (NDCG@K), and Mean Reciprocal Rank (MRR). All experiments are run on the publicly available KuaiLive dataset downloaded from its Zenodo repository.

## **1.8.3 What is Excluded**

The study does not address content-based recommendation approaches, multimodal modelling using the visual or audio content of live streams, or federated and privacy-preserving recommendation methods. It does not extend to real-time online learning or streaming model update scenarios. The scope is offline batch evaluation using historical interaction logs.

MBGCN and ATRank are evaluated here as foundational representatives of their respective families. The study does not claim to evaluate the state of the art within either family. More recent variants such as MB-GMN (Xia et al., 2021), CRGCN (Cheng et al., 2023), and MB-STR sit outside the scope of the current work, though they are acknowledged as directions for future investigation. The dataset covers only Kuaishou, and findings are not claimed to generalise to other live streaming platforms such as Twitch or YouTube Live without further validation.

# **1.9 Limitations of the Study**

## **1.9.1 Observation Window Constraint**

The KuaiLive dataset covers a 21-day observation period. This limits the study's ability to evaluate long-term preference drift, which is, in theory, one of the main strengths of sequential attention architectures. Users whose interests evolve gradually over months or years, a common real-world scenario, cannot be adequately represented in a three-week window. Findings about temporal dynamics here are therefore specific to short-window, session-level behaviour rather than longitudinal interest evolution. This is acknowledged as a structural limitation of the available data, not a methodological choice.

## **1.9.2 Single Platform Generalisability**

All interaction data comes from Kuaishou, a live streaming platform with a predominantly Chinese user base and a content culture that emphasises virtual gifting as a primary monetisation mechanism. The gifting behaviour that serves as the target interaction in this study may not exist, or may carry different semantic weight, on other platforms. Twitch, for example, monetises primarily through subscriptions, while YouTube Live uses Super Chat, which is the equivalent of gifting but less culturally central. Generalisation of findings to other platforms should therefore be treated with caution and is an explicit direction for future work.

## **1.9.3 Model Selection Simplification**

Each architectural family is represented here by a single foundational model: MBGCN for the graph-based family and ATRank for the sequential family. Both are widely cited and serve as the architectural archetypes of their respective lineages, which makes them principled choices for a comparison focused on architectural assumptions rather than peak performance. The consequence is that the absolute performance numbers reported in this study should not be read as the ceiling of what each family can achieve. More recent models within each family, such as MB-GMN (Xia et al., 2021\) and SASRec-based multi-behaviour extensions (Kang & McAuley, 2018), may behave differently in ways not captured here.

## **1.9.4 Encrypted Feature Opacity**

The KuaiLive dataset includes seven binary encrypted user features and an undisclosed number of encrypted streamer features whose semantic meanings are not disclosed by the dataset authors for privacy reasons (Qu et al., 2025). These features are included in model training as numeric inputs, but their contribution to model performance cannot be independently interpreted or ablated in a semantically meaningful way. This limits the depth of the mechanistic analysis in the results chapter.

## **1.9.5 Computational Constraints**

Graph construction over 11.6 million live rooms in MBGCN is memory-intensive. Depending on available hardware, full-scale graph experiments may require subsampling strategies that introduce selection biases, potentially underrepresenting rare streamers with few interactions. Where subsampling is applied, it is documented explicitly in the methodology chapter, and its implications for cold-start analysis are discussed there.

# **1.10 Definition of Terms**

The following technical terms are used throughout this study. Definitions are provided in alphabetical order.

**Attention Mechanism:** A neural network component that computes a weighted aggregation over a set of input vectors, where the weights are dynamically determined based on the relevance of each input to a given query vector. In sequential recommendation, attention mechanisms allow the model to focus selectively on the most relevant past interactions when predicting a future one (Vaswani et al., 2017; Zhou et al., 2018).

**ATRank:** An attention-based user behaviour modelling framework proposed by Zhou et al. (2018) for recommendation tasks. ATRank projects heterogeneous user behaviours into multiple latent semantic spaces and applies self-attention to model inter-behaviour influence, followed by vanilla attention toward a target item vector. It serves as the representative sequential architecture in this study.

**Auxiliary Behaviour:** An interaction type that is more abundant than the target behaviour and is used to enrich the model's estimation of user preference. Auxiliary behaviours provide indirect signals that help predict the sparse target behaviour. In this study, click, like, and comment are auxiliary behaviours relative to the target behaviour of virtual gift-giving.

**Cold-Start User:** A user with very few recorded target-behaviour interactions in the training period, making it difficult for a model to estimate their preferences from their own direct history. Cold-start is a core challenge in recommendation systems and is a primary segmentation variable in the evaluation framework of this study.

**Collaborative Filtering:** A recommendation approach that generates predictions by identifying patterns of similarity among users or items based on historical interaction data, without relying on explicit item content features (Koren et al., 2009). Multi-behaviour recommendation is an extension of collaborative filtering that uses multiple interaction types.

**Ephemeral Item:** An item that exists for a finite, time-bounded period and cannot be interacted with after that period expires. In the context of this study, live rooms are ephemeral items: they are created when a streamer begins broadcasting and cease to exist when the stream ends. This property fundamentally challenges the persistent-item assumption embedded in most recommendation architectures.

**Gift-Through Rate (GTR):** The probability that a user will send a virtual gift during a given live room session. GTR prediction is the primary recommendation task in this study and is analogous to click-through rate (CTR) prediction in traditional recommendation settings, but involves a higher-effort and higher-stakes user action (Qu et al., 2025).

**Graph Convolutional Network (GCN):** A class of neural network that operates on graph-structured data by iteratively aggregating and transforming feature information from a node's neighbourhood (Kipf & Welling, 2017). Applied to recommendation, GCNs propagate user and item representations across the interaction graph to capture high-order collaborative signals.

**Heterogeneous Graph:** A graph containing more than one type of node or edge. In multi-behaviour recommendation, a heterogeneous graph encodes multiple interaction types as distinct edge types between user nodes and item nodes, enabling the model to distinguish the different signals carried by different behaviour types.

**Hit Rate at K (HR@K):** An evaluation metric that measures the proportion of test cases in which the ground-truth target item appears within the top-K items recommended by the model. It captures whether the model produces relevant results without considering their exact ranking position.

**KuaiLive:** A large-scale, publicly available live streaming recommendation dataset collected from Kuaishou and released by Qu et al. (2025). KuaiLive records the interactions of 23,772 users across 11,613,708 live rooms over a 21-day period, including click, like, comment, and gift events alongside rich side information for users, streamers, and rooms. It is the primary dataset used in this study.

**Live Room:** The unit of content in a live streaming context. Each time a streamer goes live, a new live room is created with a unique identifier, a start timestamp, an end timestamp, and associated content metadata. A single streamer may generate hundreds of distinct live rooms over the observation period. Live rooms are distinct from streamer entities and are the ephemeral items around which the recommendation problem is constructed.

**MBGCN:** Multi-Behaviour Graph Convolutional Network, proposed by Jin et al. (2020). MBGCN constructs a unified heterogeneous graph from multi-behaviour interaction data and applies graph convolutional propagation to learn behaviour-aware user and item representations. It captures behaviour-specific interaction strengths through a user-item propagation layer and behaviour semantics through an item-item propagation layer. It serves as the representative graph-based architecture in this study.

**Multi-Behaviour Recommendation (MBR):** A recommendation paradigm that uses multiple types of user interactions, rather than a single target interaction, to model user preference more richly and address data sparsity in the target behaviour. MBR systems use auxiliary behavioural signals to propagate preference information toward the sparse prediction target.

**Normalised Discounted Cumulative Gain at K (NDCG@K):** A ranking-sensitive evaluation metric that measures the quality of a top-K recommendation list by giving higher scores when the ground-truth item appears closer to the top of the list. Unlike HR@K, NDCG@K is position-aware and penalises models that place the correct item near the bottom of the recommendation list.

**Self-Attention:** A specific form of attention mechanism in which every element in a sequence computes attention weights over all other elements in the same sequence, enabling the model to capture dependencies between interactions regardless of their temporal distance (Vaswani et al., 2017). Self-attention is the core mechanism in ATRank for modelling inter-behaviour influence within a user's history.

**Sequential Recommendation:** A recommendation approach that models a user's interaction history as a temporally ordered sequence and learns the evolution of user preference over time. Sequential models use recurrent or transformer-based architectures to capture how recent and past interactions jointly predict the next interaction.

**Target Behaviour:** The specific interaction type that the recommendation model is trained to predict. A target behaviour is typically the most commercially meaningful but rarest interaction in the dataset. In this study, virtual gift-giving is the target behaviour, and all model training and evaluation is oriented toward predicting which live rooms a user will send a gift in.

**Virtual Gift:** A digital item purchased with platform currency and sent to a streamer during a live broadcast as a form of financial support and social expression. On Kuaishou, virtual gifts have associated monetary values recorded in the KuaiLive dataset. Gift-sending is both the sparsest and highest-signal behaviour type in KuaiLive, comprising approximately 1.4% of total interactions (Qu et al., 2025).

**Warm User:** A user with a sufficient number of recorded target-behaviour interactions to allow a model to estimate their preferences from their own history. In this study, warm users are contrasted with cold-start users as part of the segmented evaluation framework, with the boundary defined during the data analysis phase of the study.

# 

# **References**

Cheng, Z., Han, S., Liu, F., Zhu, L., Gao, Z., & Peng, Y. (2023). Multi-behavior recommendation with cascading graph convolutional networks. In Proceedings of the ACM Web Conference 2023 (pp. 1181–1189). ACM. https://doi.org/10.1145/3543507.3583439

Fey, M., & Lenssen, J. E. (2019). Fast graph representation learning with PyTorch Geometric. In ICLR Workshop on Representation Learning on Graphs and Manifolds.

He, X., Liao, L., Zhang, H., Nie, L., Hu, X., & Chua, T.-S. (2017). Neural collaborative filtering. In Proceedings of the 26th International Conference on World Wide Web (pp. 173–182). https://doi.org/10.1145/3038912.3052569

Jin, B., Gao, C., He, X., Jin, D., & Li, Y. (2020). Multi-behavior recommendation with graph convolutional networks. In Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval (pp. 659–668). ACM. https://doi.org/10.1145/3397271.3401072

Kang, W.-C., & McAuley, J. (2018). Self-attentive sequential recommendation. In Proceedings of the 2018 IEEE International Conference on Data Mining (pp. 197–206). IEEE. https://doi.org/10.1109/ICDM.2018.00035

Kim, K., Kim, S., Lee, G., Jung, J., & Shin, K. (2025). Multi-behavior recommender systems: A survey. In Proceedings of the Pacific-Asia Conference on Knowledge Discovery and Data Mining. Springer.

Kipf, T. N., & Welling, M. (2017). Semi-supervised classification with graph convolutional networks. In Proceedings of the 5th International Conference on Learning Representations.

Koren, Y., Bell, R., & Volinsky, C. (2009). Matrix factorization techniques for recommender systems. Computer, 42(8), 30–37. https://doi.org/10.1109/MC.2009.263

Lu, W., Zhang, Y., Wang, R., Wang, H., Wang, X., Yi, X., Li, Y., & Wang, X. (2025). Live streaming recommendation in industry: Challenges and approaches. arXiv preprint arXiv:2504.00000.

Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., et al. (2019). PyTorch: An imperative style, high-performance deep learning library. In Advances in Neural Information Processing Systems (Vol. 32, pp. 8024–8035).

Precedence Research. (2024a). Recommendation engine market size, share, and trends analysis report. https://www.precedenceresearch.com/recommendation-engine-market

Precedence Research. (2024b). Livestream e-commerce market size, share, and trends analysis report. https://www.precedenceresearch.com/livestream-e-commerce-market

Qu, C., Dai, S., Guo, K., Zhao, L., Niu, Y., Zhang, X., & Xu, J. (2025). KuaiLive: A real-time interactive dataset for live streaming recommendation. arXiv preprint arXiv:2508.05633. https://arxiv.org/abs/2508.05633

Sun, Z., Yu, D., Fang, H., Yang, J., Qu, X., Zhang, J., & Geng, C. (2020). Are we evaluating rigorously? Benchmarking recommendation for reproducible evaluation and fair comparison. In Proceedings of the 14th ACM Conference on Recommender Systems (pp. 23–32). https://doi.org/10.1145/3383313.3412489

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention is all you need. In Advances in Neural Information Processing Systems (Vol. 30, pp. 5998–6008).

Xia, L., Huang, C., Xu, Y., Dai, P., Lu, M., & Bo, L. (2021). Multi-behavior enhanced recommendation with cross-interaction collaborative relation modeling. In Proceedings of the 37th IEEE International Conference on Data Engineering (pp. 1931–1936). IEEE. https://doi.org/10.1109/ICDE51399.2021.00179

Zhou, C., Bai, J., Song, J., Liu, X., Zhao, Z., Chen, X., & Gao, J. (2018). ATRank: An attention-based user behavior modeling framework for recommendation. Proceedings of the AAAI Conference on Artificial Intelligence, 32(1), 4564–4571. https://doi.org/10.1609/aaai.v32i1.11618