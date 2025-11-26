@ray.remote(num_cpus=1)
class SampleTableManager:

    def __init__(
        self, 
        table_name: str, 
        column_name: list[str], 
        column_type: list[str]
    ) -> None:
    """
    功能描述：
    在 driver 上为每个 agent 创建一个 SampleTable 实例。

    参数：
    table_name: str
    表名，与 agent_name 对应。

    column_name: list[str]
    表的列名列表。

    column_type: list[str]
    与 column_name 一一对应的列类型字符串列表。

    返回值：
    None
    """

    def insert_samples(
        self,
        policy_version: int,
        column_name: list[str],
        sample_value: list[list[Any]],
        sample_id: Optional[list[str]] = None,
        rollout_n: int = 1,
    ) -> bool:
    """
    功能描述：
    将指定 policy_version 下的一批 sample 插入表格中。sample_id 会自动生成，格式为"{input_id}_{number_of_turns}_{trajectory_id}"。
    对于 next agent 的样本插入，为了保证 sample_id 的连贯性，需显式填入 sample_id（由 last agent 生成）。

    参数:
    policy_version: int
    这批样本对应的策略版本。

    column_name: list[str]
    本次插入的列名列表（必须是初始化时声明过的列子集）。

    sample_value: list[list[Any]]
    二维列表，外层长度为样本条数，内层长度等于 len(column_name)，表示每个样本在各列上的值。

    sample_id: Optional[list[str]]（默认 None）
    如果提供，则显式指定每条样本的 sample_id；否则自动生成。长度需与 sample_value 的样本数一致。

    rollout_n: int（默认 1）
    针对 GRPO 算法，每条样本复制的份数，当 rollout_n > 1 时，会对每个样本复制多份，每份有不同的 sample_id，由其中的 trajectory_id 区分。

    返回值：
    bool：True：所有样本插入成功（包括写入后端存储并更新元数据表）；False：存在至少一条样本或一列写入失败
    """

    def retrieve_sample_columns(
        self,
        policy_version: int,
        column_name: list[str],
        batch_size: int = -1,
        condition: Optional[str] = None,
    ) -> tuple[list[str], dict[str, list[Any]]]:
    """
    功能描述：
    在指定 policy_version 下，按照 condition 条件过滤样本行，返回 batch_size 个样本对应的指定列的值。

    参数：
    policy_version: int
    提取指定策略版本的 sample。

    column_name: list[str]
    需要提取的列名列表。

    batch_size: int（默认 -1）
    返回的样本条数；batch_size < 0 表示返回所有符合条件的样本；batch_size > 0 表示返回 batch_size 条。

    condition: str（默认 None）
    过滤条件字符串，例如 "reward_status = True and reward > 0"。实现中会当成一个布尔表达式，可使用的变量为所有列名。

    返回值：
    sample_ids: list[str]
    按顺序返回的样本 ID 列表。

    sample_values: dict[str, list[Any]]
    形如 {column_name: [v1, v2, ...]} 的字典，每个列名对应一个与 sample_ids 对齐的值列表。
    """

    def write_sample_columns(
        self,
        sample_id: list[str],
        column_name: list[str],
        sample_value: dict[str, list[Any]],
    ) -> bool:
    """
    功能描述：
    根据 sample_id 和 column_name 为已有样本写入或覆盖源数据。

    参数：
    sample_id: list[str]
    需要写入 / 更新的样本 ID 列表。

    column_name: list[str]
    需要写入 / 更新的列名列表。

    sample_value: dict[str, list[Any]]
    sample_value[col][i] 对应第 i 个 sample_id 在列 col 上的值。

    返回值：
    bool：是否所有写入都成功。
    """

    def delete_samples(
        self, 
        policy_version: int = -1, 
        condition: Optional[str] = None,
    ) -> bool:
    """
    功能描述：
    删除指定策略版本下，满足条件 condition 的所有行的元数据（SampleTable 中的记录）以及对应的源数据（后端存储中的记录）。
    策略版本和 condition 至少需要指定一个。

    参数：
    policy_version: int（默认 -1）。大于 0：只删除该 policy_version 下的样本；小于 0：不根据策略版本过滤（即对所有策略版本生效）。

    condition: str（默认 None）
    过滤条件字符串，与 retrieve_sample_columns 中语义一致。空字符串：删除指定 policy_version 的所有样本；非空：仅删除满足条件的样本。

    返回值：
    bool：True：所有元数据行及其对应源数据都成功删除；False：存在删除失败的情况。
    """

    def __retrieve_data(
        self,
        sample_id: str,
        column_name: str,
    ) -> Any:
    """
    功能描述：
    根据 sample_id 和 column_name 从后端存储系统读取 sample_value，用于实现 condition 条件判断。

    参数：
    sample_id: str
    单个样本的 ID。

    column_name: str
    单个列名。

    返回值：
    sample_value: Any
    读取的数据值。
    """

    def __write_data(
        self,
        sample_id: str,
        column_name: str,
        sample_value: Any,
    ) -> tuple[bool, str]:
    """
    功能描述：
    逐条将sample (sample_id, column_name, sample_value) 写入后端存储系统，并返回写入状态和位置（key）。
    policy_version、sample_id 以及 status 字段直接存储在表格中。

    参数：
    sample_id: str
    单个样本的 ID。

    column_name: str
    单个列名。

    sample_value: Any
    写入的数据值。

    返回值： (ok: bool, location: str)
    ok：写入是否成功；location：数据在后端存储系统中的 key。
    """
