# Annotation Guidelines

These guidelines summarize the rules used to annotate the corpus (paper §3.2–§3.4).
The schema (types and definitions) is in [`schema/schema.json`](../schema/schema.json);
the annotation tool configuration is in [`schema/label_studio_config.xml`](../schema/label_studio_config.xml).

## 1. Unit of annotation

- A **document** is a curated passage from one open-access article: the **abstract plus two paragraphs** from the first half of the article (paper §3.1.1).
- Entities and relations are annotated over the **whole passage**. Relations can cross sentence and paragraph boundaries.
- Figures, tables, and equations are excluded from the text.

## 2. General principles

1. **Read the whole text first**, then mark entity boundaries and labels.
2. **Explicit evidence only.** Assign a label or relation only when the text supports it. Do not use outside knowledge, inference, or mere co-occurrence.
3. **One label per span.** Each annotated span gets exactly one of the 11 entity types.
4. **No overlapping spans.** Nested or overlapping spans are not part of the schema. Any that remain are dropped during preprocessing.
5. **The same schema in every domain.** The 11 entity types and 5 relation types are applied uniformly across all 11 domains.

## 3. Entity types

| Label | Definition | Examples from the corpus |
|---|---|---|
| `Subject` | A problem, challenge, or field that is the focus of study or problem-solving. | *climate change*, *energy poverty*, *smart grid* |
| `Theory` | Principles or theories that describe, explain, or predict particular phenomena. | *Poisson equation*, *Laplace equation*, *Thomson effect* |
| `Criteria` | Standards or indicators used to evaluate and judge quality. | *30-day mortality*, *Charlson comorbidity index*, *figure of merit* |
| `Policy` | Principles, rules, or guidelines that direct decisions and actions. | *climate change mitigation policies*, *Demand-side Management (DSM)*, *Sustainable Development Goals* |
| `Material` | A physical or chemical substance or compound with defined properties, used in processes or in producing products. | *MR fluid*, *culture medium*, *functionally graded MMCs* |
| `Physical_Tools` | Tools and systems of a material nature used to carry out specialized activities. | *hydrophone*, *valve*, *TEG device* |
| `Non_Physical_Tools` | Non-material tools and systems (e.g., software) used to carry out specialized activities. | *reader software*, *Smart Energy Management Systems*, *DeblurGAN* |
| `Data` | Datasets defined for a specific task. | *real estate pricing data*, *reconstructed measurement data* |
| `Process` | A structured sequence of steps aimed at achieving a specific goal or producing a particular output. | *photofermentation*, *dissociative attachment*, *recombination* |
| `Method` | A systematic approach or technique for analysis, problem-solving, or achieving a goal. | *Classical Trajectory Monte Carlo Method*, *Dirac-Hartree-Fock methodologies* |
| `Parameter` | A property, value, or variable that defines or controls the behavior of a system, model, or process. | *angle of attack*, *magnetic field intensity*, *pressure* |

### Distinctions that are often confused

These pairs account for most annotator disagreement and model errors (paper §4.2.1):

- **Method or Process:** a *Method* is an approach or technique. A *Process* is a sequence of steps or events that produces an output.
- **Physical_Tools or Non_Physical_Tools:** use *Physical_Tools* for material devices and hardware. Use *Non_Physical_Tools* for software, algorithms used as tools, platforms, and systems with no physical form.
- **Non_Physical_Tools or Method:** a named software package or system is a tool. The technique it implements is a method.
- **Material or Physical_Tools:** a substance is a *Material*. A device made from it is a *Physical_Tools* entity.
- **Criteria or Parameter:** use *Criteria* only when the quantity is used to **evaluate or judge** something. Otherwise it is a *Parameter*.
- **Data or Parameter:** *Data* is a dataset or a body of measurements for a task. A single variable is a *Parameter*.

## 4. Relation types

All relations are **directed**: `head → tail`.

| Label | Definition | Reading |
|---|---|---|
| `PART_OF` | One entity is a part of another (part–whole). | *head* is part of *tail* |
| `CREATES` | One entity builds or produces another entity. | *head* creates *tail* |
| `STUDIES` | One entity examines, analyzes, or studies another entity. | *head* studies *tail* |
| `CAUSES` | One entity is explicitly the cause of another entity or phenomenon. | *head* causes *tail* |
| `RELATED_TO` | A clear semantic relation that fits none of the four types above (residual type). | *head* is related to *tail* |

Rules:

1. Relations may hold between entities of **any** type.
2. Annotate a relation only when the text **explicitly** supports it.
3. Prefer a specific type. Use `RELATED_TO` only when the relation is clear but none of the four specific types fits.
4. **Check the direction.** `(A, B)` and `(B, A)` are different relations.
5. Pairs without an annotated relation are treated as `no_relation` for training.

## 5. Workflow (paper §3.3–§3.4)

**Entity phase**
1. Human annotators label entities in Label Studio, in small batches.
2. After each batch, domain experts review a random sample so that systematic errors are corrected before work continues.
3. Most domains use overlapping annotation. Disagreements go to an LLM adjudicator (GPT-4.1), which judges each disputed case three times independently; the majority label wins.
4. A domain expert reviews and confirms the adjudicated labels.

**Relation phase**
1. GPT-4.1 and DeepSeek independently propose relations between the consolidated entities, each with a confidence score. Proposals below 50% confidence are discarded.
2. Proposals made by both models form the agreement set. Disputed proposals go through cross-review, in which each model re-evaluates the other's proposal.
3. Human annotators review **every** candidate. They correct directions, amend types, remove relations without enough evidence, and add missing relations.

**Agreement** (Krippendorff's α)
- Independent entity agreement among human annotators: mean ≈ 0.38.
- After consolidation: entities ≈ 0.73, relations ≈ 0.60.

The full per-domain values are in paper Table 4.
