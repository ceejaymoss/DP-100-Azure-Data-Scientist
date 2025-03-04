#
# This code is used in Azure monitor
AzureDiagnostics
    take 5
#
# Generates the latest 5
AzureDiagnostics
    top 5 by TimeGenerated
#
# Search for a specific piece of text
Search in (AzureDiagnostics) "Response"
    Top 5 by TimeGenerated
#
#
AzureDiagnostics
    where Resource =~ "Request"
    Top 5 by TimesGenerated
#
#
AzureDiagnostics
    where Resource =~ "Request"
    project TimeGenerated, Resource, status_s
#
# Counts total logs
AzureDiagnostics
    summarize count()
#
# Creates a column chart showing succeeds and running
AzureDiagnostics
    summarize event_count=count() by status_s
    render columnchart
#
#
AzureDiagnostics
    where Resource =~ "response"
    where status_s =~ "succeeded"
#
#
AzureDiagnostics
    where Resource =~ "Response"
    project TimeGenerated, Resource, status_s
#
#
AzureDiagnostics
    summarize count() by status_s
#
#