# Project Explanation

## Project Title

AI Based Voice Stress Analysis System for Lie Possibility Detection

## What the Project Does

This project analyzes English voice answers and estimates Lie Possibility. The system checks voice stress, pauses, hesitation, speaking speed, changes from normal voice, and answer matching across multiple responses.

The result is shown as:

```text
Lie Possibility: Low
Lie Possibility: Medium
Lie Possibility: High
```

The system does not say that a person is lying. It only gives an estimate based on the recorded voice and answer pattern.

## Why Normal Voice Sample Is Used

Every person has a different speaking pattern. Some people naturally speak fast. Some people take pauses. Some people have a low or high voice pitch.

The normal voice sample gives the system a reference point. Later answers are compared with that normal voice sample. If the answer voice changes more than expected, the system adds that to the Lie Possibility score.

## Updated Test Flow

```text
Normal Voice Sample
↓
Initial Statement
↓
Adaptive Follow-up Questions
↓
Analysis
↓
Final Result
```

The user is not forced to answer only one fixed story such as going somewhere yesterday. The user first gives a statement about any activity or situation. The system then creates follow-up questions from that statement.

Example:

```text
Initial statement: I stayed at home yesterday and studied Python.
```

Generated questions may be:

```text
What time did you start studying?
Which subject or topic were you studying?
Where were you studying?
What time did you stop studying?
```

If the statement is about travel, shopping, work, or staying at home, the questions change according to the statement.

## How Voice Answers Are Checked

The system extracts useful audio features from each voice answer:

- Duration
- Energy level
- Voice stability
- Silence ratio
- Estimated pause count
- Pitch estimate
- Spectral behavior
- MFCC mean values

These values are used internally. The app UI shows only important labels such as:

- Voice Stress
- Pause Level
- Hesitation
- Answer Matching
- Speaking Speed
- Voice Change

## How Speech-to-Text Works

The project uses faster-whisper for English speech-to-text. When the user records or uploads an answer, the system tries to create a transcript automatically.

If speech-to-text cannot create a transcript because of unclear audio or missing model files, the app allows transcript correction or manual entry. This prevents the test from stopping completely.

## How Adaptive Questions Work

The file `src/question_planner.py` reads the first statement and checks for clues such as:

- Activity
- Place
- Time
- People
- Study-related words
- Work-related words
- Travel-related words
- Purchase-related words

Based on these clues, it creates follow-up questions that match the user's statement.

This improves the project because a fixed question set may not match every user. For example, if the user says they stayed at home, travel questions do not make sense. Adaptive questions solve that issue.

## How Answer Matching Works

The file `src/answer_matching.py` checks the transcript text. It looks for:

- Very short answers
- Repeated hesitation words
- Repeated words
- Time review points
- Unclear statements
- Activity changes between the first statement and later answers

The app shows the matching result as:

```text
Matched
Not Fully Matched
```

The app does not show heavy technical wording in the visible UI.

## Scoring Formula

The final score uses a hybrid method:

```text
30% Voice Stress
25% Pause and Hesitation
25% Difference from Normal Voice
20% Answer Matching
```

The score is mapped like this:

```text
0% - 35% = Low Lie Possibility
36% - 65% = Medium Lie Possibility
66% - 100% = High Lie Possibility
```

## Why the Result Is Shown as Possibility

Voice analysis cannot prove a lie with full certainty. Stress, fear, poor microphone quality, background noise, illness, and nervousness can change a person's voice.

That is why the project uses the wording Lie Possibility. It gives a careful estimate instead of making a final claim.

## Teacher Explanation

This project is an AI based voice analysis system. The user first records a normal voice sample. After that, the user answers a set of questions using voice. The system checks voice stress, pauses, hesitation, speaking speed, and answer matching. Based on these checks, the system shows a lie possibility result as Low, Medium, or High. The result is only an estimate and does not prove that a person is lying.

## What Makes This Version Stronger

- It uses an English speech-to-text pipeline.
- It has adaptive questions based on the user's first statement.
- It does not depend on one fixed story.
- It has microphone recording and audio upload fallback.
- It compares answers against a normal voice sample.
- It uses a hybrid scoring method instead of depending only on one ML model.
- It includes a downloadable PDF report.
- It avoids unsafe claims about lie detection.

## Expected Questions and Answers

### Q1. Does this project detect lies with 100% accuracy?

No. The system does not claim 100% lie detection. It estimates lie possibility using voice stress and answer matching.

### Q2. Why do we record a normal voice sample?

The normal voice sample helps the system compare the user's usual speaking pattern with the answers given during the test.

### Q3. What signs does the system check?

The system checks voice stress, pauses, hesitation, speaking speed, voice changes, and whether the answers match properly.

### Q4. Why is the result shown as Low, Medium, or High?

Because voice analysis can only estimate possibility. It should not be treated as final proof.

### Q5. What is the use of this project?

This project can be used as a support tool for voice behavior analysis, interview practice, investigation training, and research-based presentations.

### Q6. Why are the questions adaptive?

The first statement can be about any situation. Adaptive questions keep the test relevant by asking follow-up questions based on what the user said.

### Q7. What happens if speech-to-text fails?

The app keeps the test running and allows the transcript to be entered or corrected before analysis.


## PDF Report

The PDF report includes the initial statement, the adaptive questions asked by the system, and the user answer transcripts. This makes the report more transparent because the teacher can see the spoken statement, the generated questions, the answers, and the final Lie Possibility result in one place.

The report also includes a transcript note because speech-to-text output can contain minor recognition errors.


The downloadable PDF report includes the final result, main reasons, safety note, question-answer transcripts, and a Voice Stress Level per Answer chart.
