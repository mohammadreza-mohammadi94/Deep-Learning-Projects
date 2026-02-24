import os
import re

CATEGORIES = [
    "Natural-Language-Processing",
    "Computer-Vision",
    "Time-Series-Forecasting",
    "Audio-Processing",
    "Signal-Processing",
    "Reinforcement-Learning",
    "Tabular-Data-Analysis"
]

def generate_project_list():
    lines = ["## Projects Overview\n\n"]
    for category in CATEGORIES:
        if not os.path.exists(category):
            continue
        # Format category name for display
        display_category = category.replace("-", " ")
        lines.append(f"### {display_category}\n")
        
        # List projects in category
        category_path = category
        projects = sorted([d for d in os.listdir(category_path) if os.path.isdir(os.path.join(category_path, d))])
        
        for project in projects:
            # Create a markdown link to the project directory
            # Encode spaces in the URL
            project_url = f"{category}/{project.replace(' ', '%20')}"
            lines.append(f"- [{project}]({project_url})\n")
        lines.append("\n")
    return "".join(lines)

def update_readme():
    readme_path = "README.md"
    if not os.path.exists(readme_path):
        print("README.md not found!")
        return

    with open(readme_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    project_list = generate_project_list()
    
    # Section markers to replace
    # This pattern will match either "Available Projects" or "Projects Overview"
    # and everything up to "Prerequisites"
    pattern = r"## (Available Projects|Projects Overview).*?(?=(### Prerequisites|## Prerequisites))"
    
    if re.search(pattern, content, re.DOTALL):
        new_content = re.sub(pattern, project_list, content, flags=re.DOTALL)
        print("Updated existing project list section.")
    else:
        # Fallback: try to insert it before Prerequisites
        if "### Prerequisites" in content:
            new_content = content.replace("### Prerequisites", project_list + "### Prerequisites")
            print("Inserted project list before 'Prerequisites'.")
        elif "## Prerequisites" in content:
            new_content = content.replace("## Prerequisites", project_list + "## Prerequisites")
            print("Inserted project list before 'Prerequisites'.")
        else:
            new_content = content + "\n\n" + project_list
            print("Appended project list to the end of README.md.")
            
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(new_content)

if __name__ == "__main__":
    update_readme()
