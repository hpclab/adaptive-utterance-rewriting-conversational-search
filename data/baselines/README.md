# Baselines

This directory collects files with manual utterances as well as queries/utterances rewritten automatically by the baselines used in [Mele2021].

## Manual utterance file

`manual_utterance.tsv`: utterances rewritten manually by human annotators.

## Baselines and corresponding files

- `query.tsv`:  CAsT 2019 baseline consisting of queries generated from utterances by simply applying stopword removal and AllenNLP co-reference resolution.

- `first_query.tsv`: Queries rewritten using the terms from the first-turn query (e.g., q_1 + q_i).

- `context_query.tsv`: Queries rewritten with the first query and the one appearing in the previous turn (e.g., q_1 + q_i−1 + q_i).

- `plain_utterance.tsv`: Plain Utterance provided by CAsT without performing any rewriting.

- `coRef1/2.tsv`: Rewriting methods using co-reference resolution. It finds all the linguistic expressions that refer to the same real-world entity in a natural language text. We used two different models for co-referencing: (1) the _AllenNLP_ co-referencing model [Gardner2018, Lee2017], and (2) the _neuralcoref_ model from the Transformers library, which are applied to the original utterances to produce the queries to process.

- `first_topic.tsv`: Queries rewritten using an approach originally proposed in [Mele2020]. Given a conversation, the current utterance is expanded with the main conversation topic extracted from the first-turn utterance. 

- `topic_shift.tsv`: Queries rewritten using an approach originally proposed in [Mele2020]. This strategy is similar to First Topic, but it also includes a Context on Cue step which aims at identifying context changes on the basis of some cues. The current utterance is expanded with the main conversation topic extracted either from the first-turn utterance or from the one where the topic shift occurred. 

- `context.tsv`: Queries rewritten using an approach originally proposed in [Mele2020]. This strategy is based on the previous two strategies (First Topic and Topic Shift), but it also includes the context collected with the help of a Context Binder methodology applied to all the previous utterances in the conversation. 

## Format

utteranceId \t utterance 

## References

[Mele2021] Ida Mele, Cristina Ioana Muntean, Franco Maria Nardini, Raffaele Perego, Nicola Tonellotto, Ophir Frieder, Adaptive utterance rewriting for conversational search, IProcessing & Management, Volume 58, Issue 6, 2021, https://doi.org/10.1016/j.ipm.2021.102682.

[Mele2020] Ida Mele, Cristina Ioana Muntean, Franco Maria Nardini, Raffaele Perego, Nicola Tonellotto, and Ophir Frieder. Topic propagation in conversational search. In Proc. ACM SIGIR, pages 2057–2060, New York, NY, USA, 2020. ACM.

[Gardner2018] Matt Gardner, Joel Grus, Mark Neumann, Oyvind Tafjord, Pradeep Dasigi, Nelson F. Liu, Matthew Peters, Michael Schmitz, and Luke Zettlemoyer. AllenNLP: A deep semantic natural language processing platform. In Proc. NLP-OSS, pages 1–6, Copenhagen, Denmark, 2018. ACL.

[Lee2017] Kenton Lee, Luheng He, Mike Lewis, and Luke Zettlemoyer. End-to-end neural coreference resolution. In Proc. EMNLP, pages 188–197, Copenhagen, Denmark, 2017. ACL.
