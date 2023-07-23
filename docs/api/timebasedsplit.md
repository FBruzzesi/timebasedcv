# TimeBased Cross Validation

::: timebasedcv.timebasedsplit.TimeBasedCVSplitter
    options:
        show_root_full_path: false
        show_root_heading: true

::: timebasedcv.timebasedsplit.TimeBasedSplit
    options:
        show_root_full_path: false
        show_root_heading: true

::: timebasedcv.timebasedsplit.ExpandingTimeSplit
    options:
        show_root_full_path: false
        show_root_heading: true

::: timebasedcv.timebasedsplit.RollingTimeSplit
    options:
        show_root_full_path: false
        show_root_heading: true

::: timebasedcv.timebasedsplit._CoreTimeBasedSplit
    options:
        show_root_full_path: false
        show_root_heading: true
        members:
            - _splits_from_period
            - n_splits_of
            - split
