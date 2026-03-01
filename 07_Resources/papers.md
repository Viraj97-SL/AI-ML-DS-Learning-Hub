# Essential Research Papers — Every Practitioner Should Know

> You don't need to read every paper. These are the landmark ones that shaped the field.
> Start with the abstracts. Come back to the math later.

**Difficulty:** Foundational | Intermediate | Advanced

---

## The "You Must Read These" List

### 1. Attention Is All You Need (2017)
**Authors:** Vaswani et al. (Google Brain)
**Link:** [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
**Why it matters:** Introduced the Transformer architecture. Every LLM (GPT, Claude, Gemini, Llama) is based on this. The most cited ML paper of the decade.
**Read when:** After understanding neural networks basics.

---

### 2. BERT: Pre-training of Deep Bidirectional Transformers (2018)
**Authors:** Devlin et al. (Google)
**Link:** [arXiv:1810.04805](https://arxiv.org/abs/1810.04805)
**Why it matters:** Showed that large-scale pre-training + fine-tuning dominates NLP. Sparked the transfer learning revolution.

---

### 3. Language Models are Few-Shot Learners (GPT-3) (2020)
**Authors:** Brown et al. (OpenAI)
**Link:** [arXiv:2005.14165](https://arxiv.org/abs/2005.14165)
**Why it matters:** Demonstrated emergent few-shot and zero-shot capabilities at scale. The paper that changed everything.

---

### 4. Training Language Models to Follow Instructions with Human Feedback (InstructGPT / RLHF) (2022)
**Authors:** Ouyang et al. (OpenAI)
**Link:** [arXiv:2203.02155](https://arxiv.org/abs/2203.02155)
**Why it matters:** Introduced RLHF for aligning LLMs to human intent. The technique behind ChatGPT.

---

### 5. Constitutional AI: Harmlessness from AI Feedback (2022)
**Authors:** Bai et al. (Anthropic)
**Link:** [arXiv:2212.08073](https://arxiv.org/abs/2212.08073)
**Why it matters:** Anthropic's approach to AI safety without human feedback for every step. Important for AI alignment.

---

### 6. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (RAG) (2020)
**Authors:** Lewis et al. (Facebook AI)
**Link:** [arXiv:2005.11401](https://arxiv.org/abs/2005.11401)
**Why it matters:** Introduced RAG — the dominant architecture for grounding LLMs with external knowledge. Essential for AI Engineers.

---

### 7. LoRA: Low-Rank Adaptation of Large Language Models (2021)
**Authors:** Hu et al. (Microsoft)
**Link:** [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
**Why it matters:** Made LLM fine-tuning affordable. 99% of fine-tuning done today uses LoRA or its variants.

---

### 8. Scaling Laws for Neural Language Models (2020)
**Authors:** Kaplan et al. (OpenAI)
**Link:** [arXiv:2001.08361](https://arxiv.org/abs/2001.08361)
**Why it matters:** Showed that model performance scales predictably with compute, data, and parameters. Guides how companies train big models.

---

## Deep Learning Foundations

| Paper | Authors | Year | What It Introduced |
|-------|---------|------|-------------------|
| **ImageNet Classification with Deep CNNs (AlexNet)** | Krizhevsky et al. | 2012 | Modern CNNs, deep learning revival |
| **Batch Normalization** | Ioffe, Szegedy | 2015 | Training stability, faster convergence |
| **Deep Residual Learning (ResNet)** | He et al. (Microsoft) | 2015 | Skip connections, very deep networks |
| **Dropout** | Srivastava et al. | 2014 | Regularization technique |
| **Adam Optimizer** | Kingma, Ba | 2014 | Adaptive learning rate — the default optimizer |
| **GAN: Generative Adversarial Nets** | Goodfellow et al. | 2014 | Generative models, GANs |
| **LSTM** | Hochreiter, Schmidhuber | 1997 | Long-term memory in RNNs |

---

## Transformer & LLM Papers

| Paper | Year | Key Contribution |
|-------|------|-----------------|
| Attention Is All You Need | 2017 | Transformer architecture |
| BERT | 2018 | Bidirectional pre-training |
| GPT-2 | 2019 | Language generation at scale |
| GPT-3 | 2020 | Few-shot learning |
| T5: Exploring Limits of Transfer Learning | 2020 | Text-to-text unified framework |
| Chinchilla | 2022 | Compute-optimal training (Hoffmann et al.) |
| LLaMA | 2023 | Open-source large language models |
| Llama 2 | 2023 | Open-source chat + code models |
| Mistral 7B | 2023 | Efficient small LLM |
| Gemini Technical Report | 2023 | Multimodal LLM at scale |
| Claude's Model Card | 2023 | AI safety and capabilities documentation |
| GPT-4 Technical Report | 2023 | Multimodal capabilities |

---

## NLP Papers

| Paper | Year | Key Contribution |
|-------|------|-----------------|
| Word2Vec | 2013 | Word embeddings |
| GloVe | 2014 | Global word vectors |
| ELMo | 2018 | Contextual embeddings |
| Sentence-BERT | 2019 | Sentence embeddings |
| XLNet | 2019 | Permutation language modeling |
| RoBERTa | 2019 | Optimized BERT pre-training |
| BART | 2019 | Denoising auto-encoder for seq2seq |
| GPT-NeoX | 2022 | Open-source LLM (EleutherAI) |

---

## Computer Vision Papers

| Paper | Year | Key Contribution |
|-------|------|-----------------|
| AlexNet | 2012 | Deep CNN for ImageNet |
| VGG | 2014 | Very deep convolutional networks |
| Inception (GoogLeNet) | 2014 | Efficient wide networks |
| ResNet | 2015 | Deep residual networks |
| DenseNet | 2016 | Dense connections |
| YOLO | 2016 | Real-time object detection |
| ViT: An Image is Worth 16x16 Words | 2020 | Vision Transformers |
| CLIP | 2021 | Connecting text and images |
| DALL-E | 2021 | Text-to-image generation |
| Stable Diffusion (LDM) | 2022 | Latent diffusion models |
| Segment Anything (SAM) | 2023 | Universal segmentation model |

---

## ML Theory & Algorithms

| Paper | Year | Key Contribution |
|-------|------|-----------------|
| A Few Useful Things to Know About ML | Domingos | 2012 | Practical ML wisdom |
| XGBoost: A Scalable Tree Boosting System | Chen, Guestrin | 2016 | The tree boosting standard |
| Dropout | Srivastava et al. | 2014 | Regularization |
| Batch Normalization | Ioffe, Szegedy | 2015 | Training stability |
| Deep Q-Network (DQN) | Mnih et al. (DeepMind) | 2013 | Deep reinforcement learning |
| AlphaGo | Silver et al. (DeepMind) | 2016 | RL for complex games |
| Attention and Augmented RNNs | Olah, Carter | 2016 | Visualizing attention (distill.pub) |

---

## MLOps & Systems Papers

| Paper | Year | Key Contribution |
|-------|------|-----------------|
| **Hidden Technical Debt in ML Systems** | Sculley et al. (Google) | 2015 | Why ML in production is hard |
| **Machine Learning: The High-Interest Credit Card of Technical Debt** | Google | 2014 | ML systems maintenance |
| **Lessons from Deploying ML** | Uber | 2018 | Real production ML challenges |
| **MLflow: A System for the Complete ML Lifecycle** | Zaharia et al. | 2018 | ML experiment tracking |
| **Kubeflow: A Composable, Portable, Scalable ML Stack** | Google | 2018 | ML on Kubernetes |
| **Continuous Delivery for ML** | Sato et al. | 2019 | MLOps CD/CT practices |

---

## AI Engineering & LLM Papers (2023-2025)

| Paper | Key Topic |
|-------|----------|
| **RAGAS: Automated Evaluation of RAG** | RAG evaluation metrics |
| **Self-RAG** | Selective retrieval for better RAG |
| **HyDE: Hypothetical Document Embeddings** | Better RAG retrieval |
| **ReAct: Synergizing Reasoning and Acting in LLMs** | Agent architecture |
| **Toolformer** | Teaching LLMs to use tools |
| **Gorilla: LLM Connected with Massive APIs** | Tool use |
| **LLM Powered Autonomous Agents** | Agent survey (Lilian Weng) |
| **Direct Preference Optimization (DPO)** | Alternative to RLHF |
| **QLoRA: Efficient Fine-Tuning of Quantized LLMs** | Efficient fine-tuning |
| **FlashAttention** | Efficient attention computation |
| **Mixtral of Experts** | Mixture of experts architecture |
| **Chain-of-Thought Prompting Elicits Reasoning** | CoT prompting |
| **Large Language Models are Zero-Shot Reasoners** | Let's think step by step |
| **SELF-REFINE: Iterative Refinement with Self-Feedback** | LLM self-improvement |

---

## How to Read Papers Effectively

### The 3-Pass Method
1. **Pass 1 (5 min):** Read title, abstract, section headers, figures/captions, conclusion
   - Goal: Understand what the paper claims and roughly how
2. **Pass 2 (30-60 min):** Read the introduction, results, and relevant methodology
   - Goal: Understand the key contributions and experiments
3. **Pass 3 (hours):** Read everything including the math
   - Goal: Be able to reproduce or critique the paper

### Tools for Paper Discovery
- [arXiv.org](https://arxiv.org) — Preprint server for all ML papers
- [Papers With Code](https://paperswithcode.com) — Papers + code + benchmarks
- [Semantic Scholar](https://semanticscholar.org) — AI-powered paper search
- [Connected Papers](https://connectedpapers.com) — Visualize related paper graph
- [Alpha Signal](https://alphasignal.ai) — Curated top papers weekly
- [Hugging Face Papers](https://huggingface.co/papers) — Daily trending papers

---

*Back to: [Resources](.) | [Main README](../README.md)*
