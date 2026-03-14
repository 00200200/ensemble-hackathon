# Tasks for EnsembleAI Hackathon 2026

This repository contains example task submissions for the **EnsembleAI Hackathon 2026**.  
Each task has its own directory containing the necessary code and submission examples.

---

# How to Use

## Python Environment Setup

We strongly recommend using a **Python virtual environment** before installing dependencies.

### Create a virtual environment

```bash
python3 -m venv .venv
```

### Activate the environment

```bash
source .venv/bin/activate
```


### Install dependencies

```bash
pip install -r requirements.txt
```

---

# Environment Configuration

Before running the submission script, create a `.env` file in the root directory of the project.  
This file stores configuration variables required for authentication and communication with the submission server.

## Example `.env`

```
TEAM_TOKEN="mytoken"
SERVER_URL="http://149.156.182.9:6060"
```

## Variables

- **TEAM_TOKEN** – Your team authentication token provided by the hackathon organizers. It is used to authorize submissions.
- **SERVER_URL** – Base URL of the EnsembleAI submission server where results are sent.

### Security Note

Do **not** commit your `.env` file to version control.  
Keep your `TEAM_TOKEN` private.

---

# Submitting Results

To submit results for a task, run:

```bash
python3 example_submission.py
```

After submission, the server will return a **request ID**.  
This ID can be used to check the processing status of your submission.

---

# Checking Submission Status

You can check the status of a submission using the request ID:

```bash
python3 shared/get_task_status.py --request-id <id>
```

Example:

```bash
python3 shared/get_task_status.py --request-id 123456
```

This command will return the current status of the task and the score (when available).

---

# Leaderboard

The leaderboard is available at:

http://149.156.182.6/

## Public Leaderboard

- Each task has its **own leaderboard**.
- During the hackathon, the leaderboard displays the **maximum public score** achieved by each team.

## Final Score

The **final evaluation score** is determined by:

- The **private score**
- From the **last submission sent before the hackathon deadline**

This means that even if an earlier submission achieved a higher public score, the **last submitted solution** will determine your final ranking.

---

# Troubleshooting

If you notice any strange behavior or errors, please contact the **Infrastructure Team** on Discord or on-site.