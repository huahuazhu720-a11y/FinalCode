import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime
def convert_date(date_str):
    date_str = str(date_str)
    if len(date_str) == 5:  
        date_str = "0" + date_str  
    date_obj = datetime.strptime(date_str, "%m%d%y")
    return date_obj.strftime("%Y/%m/%d")

urls=pd.read_excel('TestScrape.xlsx')
results = []
exceptions=[]
for row in urls.itertuples(index=False, name=None):
    print(row)
    url = row[1]
    data=convert_date(row[0])
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
    }
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        html_content = response.text
        soup = BeautifulSoup(html_content, "lxml")
        container  = soup.find("div", attrs={"property": "content:encoded"})
        if not container:
            container  = soup.find("div",  id="week")
        if container:            
            uls = container.find_all("ul") 
            heading_order=0
            for ul in uls:
                heading = ul.find_previous_sibling()
                try:
                    heading=heading.get_text()
                except Exception:
                    continue
                    
                heading_order+=1
                lis = ul.find_all("li")
                Title_order=0
                for li in lis:
                    Title_order+=1
                    a_tag = li.find("a")
                    if a_tag:
                        try:
                            Title = a_tag.get_text(strip=True)  
                            if Title=='Click Here':
                                text = li.get_text(" ", strip=True)                            
                                Title = text.replace(a_tag.get_text(), "").strip()
                            url = a_tag.get("href")
                            results.append((data,heading,heading_order,Title,url,Title_order))
                        except Exception:
                            exceptions.append((row[0],row[1],f"error in find tilte or url"))
                            continue
                    else:
                        results.append((data,heading,heading_order,"Title not find","url not find",Title_order))
                       
        else:
            exceptions.append((row[0],row[1],'did not find container div'))              
    else:
        
        exceptions.append((row[0],row[1],'404')) 
result_df = pd.DataFrame(results, columns=['DATE','heading', 'heading_order', 'Title','url','Title_order'])
result_df.to_csv("_OEO_1.csv", index=False)
result_df = pd.DataFrame(exceptions, columns=['date', 'url','error'])
result_df.to_csv("_OEO_exceptions_1.csv", index=False)
