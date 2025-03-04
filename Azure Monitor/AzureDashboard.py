# Creates a column chart showing succeeds and running
AzureDiagnostics
    summarize event_count=count() by status_s
    render columnchart
#
#
AzureDiagnostics
    where TimeGenerated > ago (1h)
#
#