# compliance-issues-detection

## Background

Compliance teams of financial institutions are required by law to monitor employees and detect potential breaches of regulatory compliance rules. Compliance officers seek to find examples of communications where a negative attitude is expressed towards the compliance or where compliance issues, problems, and conflicts are mentioned. This particular scenario looks for the following types of phrases and their variations that include, but are not limited to:
- Compliance training was so boring! I just answered at random, what a waste of time
- We had issues with compliance before for a similar thing
- Let's speak later - compliance problems
- Compliance will have me pinned to the table
We may consider the following approaches for scoring the content as potentially containing mentions of compliance issues:
● Pre-defined lexical patterns in an employee’s communications;
● Pre-defined lexical patterns in an employee’s communications along with detected NLP features (e.g. negative sentiment or named entities);
● An end-to-end machine learning model that is trained on a large set of real or/and synthesized content to automatically detect discussions of compliance in a negative or potentially problematic way. The last option sounds attractive as it can be improved over time with user feedback, may potentially learn more complex patterns, and be adapted for a particular customer’s data without the manual intervention of an expert linguist.

## Task

Let’s assume that we already have a system that is designed to detect discussions of compliance in communications. The scenario is very simple - as our clients are very conservative, they do not want to introduce any additional lexical restrictions and stick with the following scenario:
Text(value = "compliance")
which means that all communication with word compliance will be flagged. There are other filters that are used in this scenario like the number of participants, the relationship between participants, etc. but these are out of the scope of this task.

## Data

A subset of publicly available Enron dataset. This dataset aims to emulate real email data that we receive on a day-to-day basis and that serves as the target for scenarios.

