import requests
from bs4 import BeautifulSoup
# đường dẫn đến trang web chứa thông tin về đoạn nhạc
url = 'https://www.nhaccuatui.com/playlist/bolero-vol2-nhu-hoa.qPTYimF1KVPr.html?st=4'
# sử dụng requests để lấy nội dung trang web
response = requests.get(url)
html_content = response.text
# sử dụng BeautifulSoup để phân tích nội dung trang web và tìm kiếm tên ca sĩ
soup = BeautifulSoup(html_content, 'html.parser')
artist = soup.find('div', {'class': 'name-singer'}).text
# kiểm tra xem tên ca sĩ có phải là Như Hoa không và hiển thị kết quả tương ứng
if 'Như Hoa' in artist:
    print('Đây là đoạn nhạc của ca sĩ Như Hoa')
else:
    print('Đây không phải là đoạn nhạc của ca sĩ Như Hoa')