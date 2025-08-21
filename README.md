# 💻 Developer Careers: Insights from the StackOverflow Survey

This project explores developer career patterns using data from the **StackOverflow Developer Survey**.  
The main question I wanted to answer was:  
👉 *What factors are most associated with developers earning higher salaries?*

---

## 🚀 Motivation
I’ve always wondered: does more coding experience automatically mean a higher salary?  
This project tries to uncover patterns in salary, experience, and age from real developer survey data.  

---

## 📊 Methodology
The project follows the **CRISP-DM process**:
1. **Business Understanding** → Explore career-related questions.
2. **Data Understanding** → Look at distributions, correlations, missing values.
3. **Data Preparation** → Handle missing data, create target variable.
4. **Modeling** → Train a Linear regression model to predict high-income developers.
5. **Evaluation** → Use accuracy, precision, recall, and F1 score.
6. **Deployment** → Share insights through this GitHub repo and a blog post.

---

## 🔑 Findings
- Years of experience *does* correlate with higher income — but age alone isn’t a strong predictor.  
- The model achieved decent accuracy, suggesting we can classify high earners with simple features.  
- Still, there’s a lot more nuance (skills, education, location) that likely matter.

---

## 🛠 Libraries Used
- `pandas`
- `numpy`
- `scikit-learn`

---

## 🙏 Acknowledgments
- Data source: [StackOverflow Developer Survey](https://insights.stackoverflow.com/survey)
- Inspiration: My curiosity about tech career growth.

---
