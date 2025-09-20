# Pipeline API

The `DELM` class coordinates configuration validation, experiment setup, preprocessing,
and batched extraction. Use this page to review constructor arguments and helper methods.

::: delm.delm.DELM
    options:
      show_bases: true
      show_source: false
      members:
        - __init__
        - from_yaml
        - from_dict
        - prep_data
        - process_via_llm
        - get_extraction_results
        - get_cost_summary
        - save_cost_summary
        - get_cost_summary_df
        - summarize_cost_by_provider
        - cleanup
