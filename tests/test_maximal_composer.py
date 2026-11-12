"""Unit tests for deterministic Maximal Profile Composer."""

from zaic_2026_4.profiling.maximal_composer import (
    JoinRef,
    TableContext,
    build_join_index,
    build_table_context,
    compose_maximal_profile,
)


def test_empty_column_composition() -> None:
    """Verify 100% NULL / empty column produces explicit empty status without conflicting phrases."""
    col_stats = {
        "null_count": 50,
        "distinct_count": 0,
        "data_format": "EMPTY",
        "top_10_samples": {},
        "min_val": None,
        "max_val": None,
    }
    table_ctx = TableContext(
        table_name="Users",
        row_count=50,
        is_view=False,
        is_unified_view=False,
        source_tables=(),
    )
    result = compose_maximal_profile(
        minimal_profile='Cột "ghichu" trong bảng Users là cột trống.',
        table_name="Users",
        col_name="ghichu",
        col_stats=col_stats,
        table_ctx=table_ctx,
    )

    assert "Cột rỗng 100%, không sử dụng" in result
    assert "Không NULL" not in result
    assert "Khoảng:" not in result
    assert "Tập giá trị đầy đủ:" not in result


def test_constant_column_composition() -> None:
    """Verify single-value constant column is flagged properly."""
    col_stats = {
        "null_count": 0,
        "distinct_count": 1,
        "data_format": "NUMERIC",
        "top_10_samples": {"999999999.0": 100},
        "min_val": "999999999.0",
        "max_val": "999999999.0",
    }
    table_ctx = TableContext(
        table_name="Products",
        row_count=100,
        is_view=False,
        is_unified_view=False,
        source_tables=(),
    )
    result = compose_maximal_profile(
        minimal_profile='Định mức tồn tối đa.',
        table_name="Products",
        col_name="dinh_muc_max",
        col_stats=col_stats,
        table_ctx=table_ctx,
    )

    assert "Cột hằng số (chỉ 1 giá trị duy nhất), không có ý nghĩa khi lọc/nhóm" in result
    assert "Khoảng:" not in result


def test_primary_key_candidate() -> None:
    """Verify unique key column is recognized as candidate primary key."""
    col_stats = {
        "null_count": 0,
        "distinct_count": 250,
        "data_format": "FIXED_CODE",
        "is_fixed_length": True,
        "min_str_len": 6,
        "top_10_samples": {"SP0001": 1, "SP0002": 1},
        "min_val": "SP0001",
        "max_val": "SP0250",
    }
    table_ctx = TableContext(
        table_name="Products",
        row_count=250,
        is_view=False,
        is_unified_view=False,
        source_tables=(),
    )
    result = compose_maximal_profile(
        minimal_profile='Mã hàng hóa.',
        table_name="Products",
        col_name="ma_hang",
        col_stats=col_stats,
        table_ctx=table_ctx,
    )

    assert "Định dạng: FIXED_CODE, độ dài 6 ký tự" in result
    assert "Không NULL (250 dòng)" in result
    assert "250 giá trị duy nhất (ứng viên khóa chính)" in result
    assert "Khoảng: SP0001 → SP0250" in result


def test_repeated_identifier_count_distinct_warning() -> None:
    """Verify repeated code identifier triggers explicit COUNT(DISTINCT) guidance."""
    col_stats = {
        "null_count": 0,
        "distinct_count": 500,
        "data_format": "FIXED_CODE",
        "is_fixed_length": True,
        "min_str_len": 8,
        "top_10_samples": {"ORD00001": 4, "ORD00002": 3},
        "min_val": "ORD00001",
        "max_val": "ORD00500",
    }
    table_ctx = TableContext(
        table_name="OrderDetails",
        row_count=2000,
        is_view=False,
        is_unified_view=False,
        source_tables=(),
    )
    result = compose_maximal_profile(
        minimal_profile='Mã đơn hàng chi tiết.',
        table_name="OrderDetails",
        col_name="ma_don",
        col_stats=col_stats,
        table_ctx=table_ctx,
    )

    assert "500 giá trị khác nhau; mã lặp lại trên nhiều dòng (dùng COUNT(DISTINCT ...) khi đếm)" in result


def test_full_enumeration_and_dominant_ratio() -> None:
    """Verify distinct <= 10 shows full value domain and >= 90% triggers dominant value notice."""
    col_stats = {
        "null_count": 0,
        "distinct_count": 3,
        "data_format": "STRING",
        "top_10_samples": {"Xong": 950, "Huy": 40, "Dang_giao": 10},
    }
    table_ctx = TableContext(
        table_name="Orders",
        row_count=1000,
        is_view=False,
        is_unified_view=False,
        source_tables=(),
    )
    result = compose_maximal_profile(
        minimal_profile='Trạng thái đơn hàng.',
        table_name="Orders",
        col_name="trang_thai",
        col_stats=col_stats,
        table_ctx=table_ctx,
    )

    assert "Tập giá trị đầy đủ: 'Xong' (950), 'Huy' (40), 'Dang_giao' (10)" in result
    assert "Giá trị áp đảo: 'Xong' (chiếm 95%)" in result


def test_unified_view_notice() -> None:
    """Verify unified VIEW context is prepended immediately after minimal profile."""
    col_stats = {
        "null_count": 0,
        "distinct_count": 50,
        "data_format": "STRING",
        "top_10_samples": {"A": 10},
    }
    table_ctx = TableContext(
        table_name="BanHang_TongHop",
        row_count=1000,
        is_view=True,
        is_unified_view=True,
        source_tables=("BanHang_2025", "BanHang_2026"),
    )
    result = compose_maximal_profile(
        minimal_profile='Chi tiết bán hàng.',
        table_name="BanHang_TongHop",
        col_name="ten_khach",
        col_stats=col_stats,
        table_ctx=table_ctx,
    )

    assert "Bảng thuộc VIEW hợp nhất 2 bảng phân vùng (BanHang_2025, BanHang_2026); ưu tiên truy vấn VIEW này" in result


def test_build_join_index_filtering() -> None:
    """Verify build_join_index suppresses spurious business date joins and retains calendar joins."""
    db_profile = {
        "tables": {
            "BanHang": {
                "row_count": 1000,
                "columns": {
                    "Ngay_dat": {"data_format": "ISO_DATE", "null_count": 0, "distinct_count": 300},
                    "Ma_NV": {"data_format": "FIXED_CODE", "null_count": 0, "distinct_count": 10},
                },
            },
            "DonNhap": {
                "row_count": 500,
                "columns": {
                    "Ngay_nhap": {"data_format": "ISO_DATE", "null_count": 0, "distinct_count": 200},
                },
            },
            "calendar": {
                "row_count": 365,
                "columns": {
                    "date": {"data_format": "ISO_DATE", "null_count": 0, "distinct_count": 365},
                },
            },
            "NhanVien": {
                "row_count": 10,
                "columns": {
                    "Ma_NV": {"data_format": "FIXED_CODE", "null_count": 0, "distinct_count": 10},
                },
            },
        },
        "undocumented_joins": [
            # Spurious date join between business tables -> should be dropped
            {"col1": "BanHang.Ngay_dat", "col2": "DonNhap.Ngay_nhap", "resemblance": 1.0},
            # Real date join with calendar -> should be kept
            {"col1": "BanHang.Ngay_dat", "col2": "calendar.date", "resemblance": 0.85},
            # Real FK join with unique target -> should be kept and flagged as key
            {"col1": "BanHang.Ma_NV", "col2": "NhanVien.Ma_NV", "resemblance": 1.0},
        ],
    }

    join_index = build_join_index(db_profile)

    # BanHang.Ngay_dat should only link to calendar.date, NOT DonNhap.Ngay_nhap
    bh_date_joins = join_index.get("BanHang.Ngay_dat", [])
    targets = [j.target_col for j in bh_date_joins]
    assert "calendar.date" in targets
    assert "DonNhap.Ngay_nhap" not in targets

    # BanHang.Ma_NV -> NhanVien.Ma_NV (NhanVien.Ma_NV is unique 10/10)
    nv_joins = join_index.get("BanHang.Ma_NV", [])
    assert len(nv_joins) == 1
    assert nv_joins[0].target_col == "NhanVien.Ma_NV"
    assert nv_joins[0].is_target_unique is True

    # Check join composition string
    composed = compose_maximal_profile(
        minimal_profile="Mã nhân viên bán.",
        table_name="BanHang",
        col_name="Ma_NV",
        col_stats=db_profile["tables"]["BanHang"]["columns"]["Ma_NV"],
        table_ctx=build_table_context(db_profile)["BanHang"],
        joins=nv_joins,
    )
    assert 'Ứng viên Join: "NhanVien"."Ma_NV" (tương đồng 1.0, khóa)' in composed


def test_determinism() -> None:
    """Verify compose_maximal_profile returns identical string on repeated executions."""
    col_stats = {
        "null_count": 5,
        "distinct_count": 20,
        "data_format": "CURRENCY",
        "currency_subtype": "VND_TEXT",
        "min_val": "10,000",
        "max_val": "500,000",
        "top_10_samples": {"10,000": 30, "20,000": 25, "50,000": 15},
    }
    table_ctx = TableContext(
        table_name="Sales",
        row_count=100,
        is_view=False,
        is_unified_view=False,
        source_tables=(),
    )
    joins = [JoinRef(target_col="Prices.cost", resemblance=0.92, is_target_unique=False)]

    res1 = compose_maximal_profile("Đơn giá.", "Sales", "price", col_stats, table_ctx, joins)
    res2 = compose_maximal_profile("Đơn giá.", "Sales", "price", col_stats, table_ctx, joins)
    assert res1 == res2
