# Moderation comparison
A Python script that creates data and graphs for moderation comparisons.
This script requires Python 3.7 or higher.


## Installation
This script uses the following packages, which can be quickly installed as shown below.
- numpy
- matplotlib
- pandas
- openpyxl

```
pip3 install numpy matplotlib pandas openpyxl
```

## Usage
First configure the `config.ini` file. Then run the `project_analysis.py` script.

```
python3 project_analysis.py
```


## The plots
This script creates 8 tables/graphs and 1 output excel file with all data.
### Linear  supervisors vs moderator plot
A linear plot of supervisors vs moderator with an automated grey best fit line. Each marker represents a student, color coded according to their final grade. The markers' size scales according to the number of students that achieved the same score.

### Class change after moderation
Table showing the number students that changed to a specific class due to moderation
For all data, check excel sheet "Class change after moderation"

### Mark overview, Mark overview bin
Table showing the final mark overview
Bin plot of final achieved marks 
For all data, check excel sheet "Mark overview"

### Moderation mark change bin
Bin plot of marks that were moderated by moderator

### Moderator max min
Table showing each moderator's maximum moderation up and down
For all data, check excel sheet "Mod max min"

### Moderator moderation table
Table of moderator and supervisor pair filtered according to `filter1` in `config.ini`.

E.g. `filter1 = 5`. All moderator and supervisor pair where `moderator-supervisor >=5` would be shown.
For filtered data, check excel sheet "Mod Sup moderation diff"
For all data, check excel sheet "Mod Sup diff filtered"

### Moderator moderation table stats
Table of moderator and supervisor pair statistics filtered according to `filter2` in `config.ini`.
Average modulation computes the mean of modulated mark for moderator supervisor pair.
Average absolute modulation computes the mean then absolute of the modulated mark.
Absolute average modulation computes the absolute then mean of the modulated mark.

E.g. `filter2 = 2`. All moderator and supervisor pair where `"average absolute modulation" or "absolute average modulation" >=2` would be shown.
For filtered data, check excel sheet "Mod Sup moderation diff"
For all data, check excel sheet "Mod Sup moderation diff filter"


