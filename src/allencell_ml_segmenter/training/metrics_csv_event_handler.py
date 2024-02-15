from watchdog.events import FileSystemEvent, FileSystemEventHandler
from time import localtime, strftime
from pathlib import Path
from csv import DictReader


class  MetricsCSVEventHandler(FileSystemEventHandler):
    def __init__(self, target_path: Path):
        super().__init__()
        self._target_path: Path = target_path
    
    def _get_latest_epoch(self) -> int:
        if not self._target_path.exists():
            return 0
        
        latest: int = 0
        with self._target_path.open("r", newline="") as fr:
            dict_reader: DictReader = DictReader(fr)
            for row in dict_reader:
                latest = int(row["epoch"]) if  int(row["epoch"]) > latest else latest

        return latest

    def on_any_event(self, event: FileSystemEvent) -> None:
        if self._target_path.exists() and self._target_path.samefile(event.src_path):
            with open('/Users/daniel.saelid/Orgs/AllenCell/log.txt', 'a') as fw:
                fw.write(f'epoch: {self._get_latest_epoch()}')
    
    """
    def  on_modified(self,  event: FileSystemEvent) -> None:
         with open('/Users/daniel.saelid/Orgs/AllenCell/log.txt', 'a') as fw:
            fw.write(f'time: {strftime("%a, %d %b %Y %H:%M:%S +0000", localtime())} event type: {event.event_type} path : {event.src_path}\n')
    def  on_created(self,  event):
         with open('/Users/daniel.saelid/Orgs/AllenCell/log.txt', 'a') as fw:
            fw.write(f'time: {strftime("%a, %d %b %Y %H:%M:%S +0000", localtime())} event type: {event.event_type} path : {event.src_path}\n')
    def  on_deleted(self,  event):
         with open('/Users/daniel.saelid/Orgs/AllenCell/log.txt', 'a') as fw:
            fw.write(f'time: {strftime("%a, %d %b %Y %H:%M:%S +0000", localtime())} event type: {event.event_type} path : {event.src_path}\n')
    """