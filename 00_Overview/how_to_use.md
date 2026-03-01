# How to Use This Repository

Welcome! This guide will help you navigate and get the most out of this learning hub, regardless of your background or experience level.

---

## Step 1 — Figure Out Where You Are

### Complete Beginner (No coding experience)

If you've never written a line of code:

1. Start with **[Python Basics](../04_Foundations/programming/python_basics.md)** — 2-3 weeks
2. Learn **[SQL Fundamentals](../04_Foundations/programming/sql_basics.md)** — 1-2 weeks
3. Explore **[Math & Statistics Overview](../04_Foundations/mathematics/)** — 3-4 weeks in parallel
4. Then pick your role track

### Some Coding Experience

If you know Python basics but are new to data/ML:

1. Do a quick **[Python for Data Science refresher](../04_Foundations/programming/python_for_ds.md)**
2. Jump into the **Beginner** section of your chosen track
3. Don't skip the math foundations — they'll become important

### Software Engineer Transitioning

If you're a developer moving into ML/AI:

1. You likely know Python and Git — skip to track-specific content
2. Focus on **[Math & Statistics](../04_Foundations/mathematics/)** if it's been a while
3. Start at **Intermediate** level in your chosen track

### Experienced Data Scientist

If you have DS experience and want to expand:

1. Check the **[Role Comparison](role_comparison.md)** to identify gaps
2. Jump directly to **Intermediate or Advanced** sections
3. Use the **[Skills Checklists](../01_Data_Scientist/skills_checklist.md)** to identify blind spots

---

## Step 2 — Choose Your Track

Read the **[Role Comparison Guide](role_comparison.md)** then choose:

```
┌─────────────────────────────────────────────────┐
│  Do you love analyzing data and telling stories? │
│  → Data Scientist Track                          │
│                                                  │
│  Do you love building reliable software systems? │
│  → ML Engineer Track                             │
│                                                  │
│  Do you love building AI-powered products fast?  │
│  → AI Engineer Track                             │
└─────────────────────────────────────────────────┘
```

**You don't have to commit forever** — many people start in one track and migrate. The fundamentals transfer.

---

## Step 3 — Follow the Roadmap

Each track has a structured path:

```
Beginner → Intermediate → Advanced → Projects
```

**Within each level, the pattern is:**

```
1. Read the overview (README.md in each section)
2. Work through the learning materials in order
3. Complete the exercises/notebooks
4. Build the suggested mini-project
5. Check off items on the Skills Checklist
6. When 80%+ of the checklist is done → move to next level
```

---

## Step 4 — The Golden Rules

### Do's
- **Build projects as you go** — passive reading doesn't stick
- **Use Jupyter notebooks** — run every code example yourself
- **Take notes** — summarize concepts in your own words
- **Time-box topics** — spend 2-3 days max on a concept before moving on
- **Join the community** — ask questions in forums, Discord, GitHub Discussions
- **Track your progress** — check off items, celebrate milestones

### Don'ts
- **Don't try to learn everything** before starting projects
- **Don't follow 5 tutorials simultaneously** — pick one path and finish it
- **Don't skip the fundamentals** — they'll haunt you later
- **Don't compare your progress** to others — everyone learns differently
- **Don't wait to be "ready" to apply** — start applying at 70% of target skills

---

## How to Use the Notebooks

Every hands-on section has Jupyter notebooks. Three ways to run them:

### Option A: Google Colab (Recommended for beginners — zero setup)
Click the "Open in Colab" button at the top of any notebook.

### Option B: GitHub Codespaces (Best for consistent environment)
Click the green "Code" button on GitHub → "Codespaces" → "Create codespace".

### Option C: Local setup
```bash
# Clone the repo
git clone https://github.com/yourusername/ai-ml-ds-hub.git
cd ai-ml-ds-hub

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter lab
```

---

## Progress Tracking

### Fork and Check Off
1. Fork this repo to your own GitHub account
2. In each section's README, there are task checklists using `- [ ]` items
3. Check them off as you complete them
4. Your forked repo becomes your public learning portfolio!

### The Learning Journal (Optional but Powerful)
Create a file `my_progress.md` in your fork and log:
- Date
- What you learned
- What you built
- What was confusing (and how you resolved it)

This becomes an incredibly powerful artifact during job interviews.

---

## How This Repo Is Structured

```
📁 Root
├── 📄 README.md              — Start here, main navigation
├── 📁 00_Overview            — Role comparison, salary, and this guide
├── 📁 01_Data_Scientist      — Full DS learning track
│   ├── 📁 beginner           — Foundations for DS
│   ├── 📁 intermediate       — Core ML skills
│   ├── 📁 advanced           — Expert-level topics
│   └── 📁 projects           — Portfolio projects
├── 📁 02_ML_Engineer         — Full MLE learning track
│   └── (same structure as DS)
├── 📁 03_AI_Engineer         — Full AIE learning track
│   └── (same structure as DS)
├── 📁 04_Foundations         — Math, statistics, Python — shared across tracks
├── 📁 05_Hands_On_Labs       — Standalone notebooks and mini-projects
├── 📁 06_Interview_Prep      — Role-specific interview questions & answers
├── 📁 07_Resources           — Books, courses, papers, tools
└── 📁 08_Career_Guide        — Salary, portfolio, resume, networking
```

---

## Suggested Weekly Schedule

### Part-time Learner (10-15 hrs/week)
```
Mon: Theory + Reading (2 hrs)
Tue: Notebook exercises (2 hrs)
Wed: Rest or light review
Thu: Continue notebook / Start project (2 hrs)
Fri: Project work (2 hrs)
Sat: New concept or tutorial (3 hrs)
Sun: Review, notes, community engagement (1 hr)
```

### Full-time Learner (40+ hrs/week)
```
Morning: New concept study (3-4 hrs)
Afternoon: Hands-on coding / project work (4-5 hrs)
Evening: Review, articles, community (1-2 hrs)
Weekend: Longer projects, competitions, catch-up
```

---

## Learning Tips from Practitioners

> **"You learn to code by writing bad code first. Don't wait for perfect — ship it."**
> — Common wisdom in the developer community

> **"The best project is the one you're actually interested in. Kaggle is fine, but a project you care about is better."**

> **"Read 1 research paper per week, even if you don't understand all of it. Understanding accumulates over months."**

> **"Get a job before you feel ready. The last 30% of skills are best learned on the job."**

---

## Getting Help

- **Stuck on a concept?** Open a GitHub Discussion in this repo
- **Found a bug?** Open a GitHub Issue
- **Want to contribute?** Read [CONTRIBUTING.md](../CONTRIBUTING.md)
- **General learning questions?** See [Communities](../07_Resources/communities.md) for where to ask

---

*Ready to start? → [Pick your track](../README.md#learning-tracks)*
