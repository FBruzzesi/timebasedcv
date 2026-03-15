from __future__ import annotations

from functools import singledispatch
from typing import TYPE_CHECKING, Any, overload

import narwhals.stable.v1 as nw

if TYPE_CHECKING:
    from narwhals.stable.v1.typing import IntoDataFrame

    from timebasedcv._typing import SeriesLike, TensorLike

    @overload
    def indexing_method(arr: nw.DataFrame[Any], mask: nw.Series) -> nw.DataFrame[Any]: ...

    @overload
    def indexing_method(arr: nw.Series, mask: nw.Series) -> nw.Series: ...

    @overload
    def indexing_method(arr: TensorLike, mask: SeriesLike[bool]) -> TensorLike: ...

    def indexing_method(arr: Any, mask: Any) -> Any: ...

else:

    @singledispatch
    def indexing_method(arr: Any, mask: Any) -> Any:  # noqa: ANN401
        """Default indexing method for arrays.

        !!! warning
            Remark that `arr` should support indexing with an array.

        Arguments:
            arr: The array-like to index.
            mask: The boolean mask to use for indexing.
        """
        if len(arr) != len(mask):
            msg = "Length of arr and mask must be equal."
            raise ValueError(msg)
        return arr[mask]

    @indexing_method.register(nw.DataFrame)
    def _nw_dataframe_indexing(df: nw.DataFrame[IntoDataFrame], mask: nw.Series) -> nw.DataFrame[IntoDataFrame]:
        """Indexing method for Narwhals DataFrames.

        Arguments:
            df: The Narwhals DataFrame to index.
            mask: The boolean mask to use for indexing.
        """
        return df.filter(mask)

    @indexing_method.register(nw.Series)
    def _nw_series_indexing(series: nw.Series, mask: nw.Series) -> nw.Series:
        """Indexing method for Narwhals Series.

        Arguments:
            series: The Narwhals Series to index.
            mask: The boolean mask to use for indexing.
        """
        return series.filter(mask)
