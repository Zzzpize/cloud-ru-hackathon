from prometheus_client import Counter

checklist_operations_total = Counter(
    'checklist_operations_total',
    'Общее количество операций с чек-листами',
    ['operation', 'status']
)