from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time
import os

class PDFWatcher(FileSystemEventHandler):
    def on_created(self, event):
        if event.src_path.endswith(".pdf"):
            print(f"📄 Neues PDF erkannt: {event.src_path}")
            os.system("python generate_train_data.py data/pdfs")
            os.system("python train_ner_model.py data")

event_handler = PDFWatcher()
observer = Observer()
observer.schedule(event_handler, path="data/pdfs", recursive=False)
observer.start()

print("📡 Warte auf neue PDFs...")
try:
    while True:
        time.sleep(10)
except KeyboardInterrupt:
    observer.stop()
observer.join()
