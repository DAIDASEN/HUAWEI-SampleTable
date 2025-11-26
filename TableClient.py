class TableClient:

    def __init__(self) -> None:

        """

        功能描述：

        按需在 driver 或 worker 上实例化 TableClient，用于通过 SampleTableManager

        管理单个样本表。



        成员变量：

          - self._table_name: Optional[str]

              当前已连接的表名；在 connect_table 成功后设置，未连接时为 None。

          - self._table_handler:

              当前表对应的 SampleTableManager named actor 句柄，由 connect_table 设置。

          - self._column_name: list[str]

              当前表的列名列表，从 SampleTableManager.get_table_column_info() 获取。

          - self._column_type: list[str]

              当前表的列类型列表，与 _column_name 一一对应。

          - self._column_value_mask: list[bool]

              当前表各列的存储方式掩码：

                True  表示按值存储（int/float/bool 等元数据列）；

                False 表示按引用存储（如 prompt/response/logits 等大数据列）。

          - self._storage_client:

              底层数据系统（如 YuanrongDataSystem）的客户端实例，

              用于根据 location_key 读写大数据列的真实值。



        返回值：

        None

        """



    def connect_table(self, table_name: str) -> bool:

        """

        功能描述：

        根据 table_name 连接到对应的 SampleTableManager named actor，并通过

        get_table_column_info() 获取该表的列信息，记录到 TableClient 的内部状态。

        连接成功后，TableClient 内部只维护这一个表的信息，后续所有操作均默认

        针对该表执行。



        典型内部流程：

          1. 使用 Ray Named Actor 拿到 SampleTableManager 句柄：

               self._table_handler = ray.get_actor(f"SampleTable-{table_name}")

          2. 调用：

               column_info = ray.get(self._table_handler.get_table_column_info.remote())

             其中 column_info 可以是：

               {

                 "column_name": [...],

                 "column_type": [...],

                 "column_value_mask": [...],

                 ...

               }

          3. 将表名与列信息写入内部状态：

               self._table_name   = table_name

               self._column_name  = column_info["column_name"]

               self._column_type  = column_info["column_type"]

               self._value_mask   = column_info["column_value_mask"]

          4. 返回 SampleTableManager 句柄。



        重要约定：

          - TableClient 在任意时刻只维护一个“当前连接表”。

          - 在调用 insert_samples / retrieve / write / delete 之前，必须先成功调用 connect_table。

          - 若多次调用 connect_table，则后一次连接会覆盖之前的连接上下文。



        参数：

        table_name: str

            需要连接的表名，前缀加上 "SampleTable-" 即与对应 SampleTableManager 的 Named Actor 名称一致。



        返回值：

        bool：

            True  表示连接成功且顺利获取列信息；

            False 表示连接失败（例如 actor 不存在或调用 get_column_info 失败）。

        """



    def insert_samples(

        self,

        policy_version: int,

        column_name: List[str],

        sample_value: List[List[Any]],

        sample_id: Optional[List[str]] = None,

        rollout_n: int = 1,

    ) -> bool:

        """

        功能描述：

        向当前已连接的表中插入一批样本。样本由 policy_version、column_name、

        sample_value 和可选的 sample_id 描述。内部会根据列类型/存储策略，将

        大数据列写入后端数据系统，并将对应的 location_key 写入 SampleTableManager；

        对元数据列则直接写入 SampleTableManager。



        对调用方提供同步语义：

          - 返回 True 表示：

              * 对应大数据列已写入后端存储系统；

              * SampleTableManager.insertSamples 已成功完成，并写入元数据表。

          - 返回 False 表示插入过程存在失败（部分或全部失败）。



        参数：

        policy_version: int

            本批样本对应的策略版本。



        column_name: list[str]

            本次插入涉及的列名列表，必须是当前表列名的子集。



        sample_value: list[list[Any]]

            二维列表，外层长度为样本条数 N，内层长度为 len(column_name)，

            表示每条样本在各列上的真实值。对于大数据列，TableClient 内部会

            将真实值写入后端数据系统，并将返回的 key 作为写入 SampleTableManager

            的值。



        sample_id: Optional[list[str]]（默认 None）

            每条样本的 sample_id 列表：

              - 若为 None，则自动生成 sample_id（基于 input_id/turn/trajectory_id）。

              - 若非 None，则长度必须与样本条数一致，用于支持跨 agent 传递样本。



        rollout_n: int（默认 1）

            针对 GRPO 等算法，每条样本复制的份数，用于自动设置 sample_id（client不需处理 sample_id 的生成）。



        返回值：

        bool：

            True  表示所有样本插入成功；

            False 表示存在至少一条样本或一列写入失败。

        """



    def retrieve_sample_columns(

        self,

        policy_version: int,

        column_name: List[str],

        batch_size: int = -1,

        condition: str = "",

    ) -> Tuple[List[str], Dict[str, List[Any]]]:

        """

        功能描述：

        从当前已连接的表中同步获取一批样本数据。

       

        内部流程：

          1) 调用 SampleTableManager.retrieveSampleColumns（异步 actor 方法），

             获取 sample_id 和各列在表中的存储值（对于大数据列为 location_key，

             对元数据列为具体数值）。

          2) 根据列类型/存储策略，对大数据列通过元戎数据系统根据 key 批量读取真实数据。

          3) 将按值存储和按引用存储的结果进行合并，返回用户期望的 Python 数据类型。



        该接口是完成所有读取操作的同步接口，适合大多数常见场景。



        参数：

        policy_version: int

            选择指定策略版本下的样本。



        column_name: list[str]

            需要读取的列名列表。



        batch_size: int（默认 -1）

            返回的样本条数：

              - batch_size <= 0：返回所有满足条件的样本；

              - batch_size > 0：返回 batch_size 条样本。



        condition: str（默认 ""）

            过滤条件字符串，例如：

              "reward_status == True and reward > 0"。

            具体语法由 SampleTableManager 内部条件解析逻辑定义。



        返回值：

        (sample_ids, sample_value)

        sample_ids: list[str]

            返回的样本 ID 列表。



        sample_value: dict[str, list[Any]]

            列名到列值列表的映射：

              - key   为 sample_id；

              - value 为该 sample 对应的值列表，长度与 column_name 对齐。

            注意，对于大数据列，返回的是从数据系统中读取的真实值，而非 location_key。

        """



    def retrieveSampleColumnKeysAsync(

        self,

        policy_version: int,

        column_name: List[str],

        batch_size: int = -1,

        condition: str = "",

    ):

        """

        功能描述：

        异步从当前已连接的表中同步获取一批样本数据。

        本接口仅返回 ObjectRef，不会在调用线程中阻塞等待结果。

        拿到 ObjectRef 的表格数据后，用户还需要调用 retrieveSampleColumnValues 获取真实数据。



        内部流程：

          - 调用 SampleTableManager.retrieveSampleColumns.remote(...)，

            得到一个 ObjectRef；

          - 该 ObjectRef 内部结果为：

              (sample_id: list[str], sample_key: dict[str, list[Any]])

            其中：

              - sample_id：样本 ID 列表；

              - sample_key：样本到“表中存储值”的映射：

                    * 对大数据列：为 location_key；

                    * 对元数据列：为具体数值列表。



        调用方使用方式示例：

          1) 立即阻塞等待：

               obj_ref = client.retrieveSampleColumnKeysAsync(...)

               sample_id, sample_key = ray.get(obj_ref)

          2) 先做别的事情，再主动拿结果：

               obj_ref = client.retrieveSampleColumnKeysAsync(...)

               ... do other work ...

               sample_id, sample_key = ray.get(obj_ref)

          3) 使用 ray.wait：

               ready, _ = ray.wait([obj_ref], timeout=0.0)

               if ready:

                   sample_id, sample_key = ray.get(ready[0])



        参数：

        policy_version: int

            策略版本过滤条件，同 retrieve_sample_columns。



        column_name: list[str]

            需要获取的列名列表。



        batch_size: int（默认 -1）

            返回的样本条数，语义同 retrieve_sample_columns。



        condition: str（默认 ""）

            过滤条件字符串，语义同 retrieve_sample_columns。



        返回值：

        obj_ref:

            Ray ObjectRef 对象，内部结果为：

              (sample_id: list[str], sample_key: dict[str, list[Any]])。

        """



    def retrieveSampleColumnValues(

        self,

        column_name: List[str],

        sample_id: List[str],

        sample_key: Dict[str, List[Any]],

    ) -> Dict[str, List[Any]]:

        """

        功能描述：

        在已经通过 retrieveSampleColumnKeysAsync 拿到 sample_id 和 sample_key

        的基础上，从元戎数据系统中读取大数据列的真实值，并与元数据列一起组装成

        完整的 sample_value。



        内部流程：

          1) 根据 self._column_type / self._column_value_mask 识别哪些列是大数据列；

          2) 对于大数据列：

                 location_keys = sample_key[col_name]

                 real_values   = self._storage_client.get(location_keys)

             替换 sample_key 中对应列的值；

          3) 对于元数据列：

                 直接沿用 sample_key[col_name] 作为返回值；

          4) 组装得到：

                 sample_value: dict[str, list[Any]]



        参数：

        column_name: list[str]

            需要提取数据的列名列表。



        sample_id: list[str]

            样本 ID 列表，用于组装返回值。



        sample_key: dict[str, list[Any]]

            从 SampleTableManager.retrieveSampleColumns 获取到的“表中存储值”：

              - 对大数据列：为 location_key 列表；

              - 对元数据列：为实际数值列表。



        返回值：

        sample_value: dict[str, list[Any]]

            列名到列值列表的映射：

              - key   为 sample_id；

              - value 为该 sample 对应的值列表，长度与 column_name 对齐。

            注意，对于大数据列，返回的是从数据系统中读取的真实值，而非 location_key。

        """



    def write_sample_columns(

        self,

        sample_id: List[str],

        column_name: List[str],

        sample_value: Dict[str, List[Any]],

    ) -> bool:

        """

        功能描述：

        根据 sample_id 和 column_name，为当前已连接表中的既有样本写入或更新

        指定列的值。对大数据列，先将真实值写入元戎数据系统并得到 location_key；

        再将 location_key 或元数据列的值写入 SampleTableManager。



        本接口对调用方提供同步语义：

          - 返回 True 表示：

              * 对应的大数据列已成功写入底层存储；

              * SampleTableManager.writeSampleColumns 已成功完成。

          - 返回 False 表示写入过程中存在失败。



        参数：

        sample_id: list[str]

            需要更新的样本 ID 列表。长度应与 sample_value 中各列的值列表长度一致。



        column_name: list[str]

            需要写入或更新的列名列表。



        sample_value: dict[str, list[Any]]

            每个 sample_id 对应一个值列表（长度与 column_name 相同）：

              - 对大数据列：为真实值（由 TableClient 内部写入数据系统，并转换为 key）；

              - 对元数据列：为直接写入 SampleTableManager 的值。



        返回值：

        bool：

            True  表示所有样本的指定列均更新成功；

            False 表示存在至少一个样本/列更新失败。

        """



    def delete_samples(

        self,

        policy_version: int = -1,

        condition: str = "",

    ) -> bool:

        """

        功能描述：

        删除当前已连接表中满足条件的样本数据，包括：

          - SampleTableManager 中对应的元数据行；

          - 元戎数据系统中大数据列对应的源数据（按 location_key 删除）。



        过滤逻辑：

          - policy_version > 0 且 condition == "":

              删除该策略版本下的所有样本；

          - policy_version > 0 且 condition 非空：

              删除该策略版本下满足 condition 的样本；

          - policy_version <= 0 且 condition 非空：

              删除所有策略版本中满足 condition 的样本；

          - policy_version <= 0 且 condition == "":

              语义为“删除整表”，一般需谨慎使用，可在实现中禁止或需要额外确认。



        对调用方提供同步语义：

          - 返回 True 表示所有目标样本及其源数据已成功删除；

          - 返回 False 表示存在至少一条样本的删除失败。



        参数：

        policy_version: int（默认 -1）

            策略版本过滤条件，具体语义见上。



        condition: str（默认 ""）

            过滤条件字符串，与 retrieve_sample_columns 中语义一致。

            可用于指定 reward=0、status=False 等条件删除。



        返回值：

        bool：

            True  表示所有目标样本和对应源数据均成功删除；

            False 表示存在删除失败的情况。

        """



    '''============以下为内部私有函数============'''

    def __retrieve_data(

        self,

        key: list[str],

    ) -> list[Any]:

    """

    功能描述：

    根据 key 从元戎数据系统中读取 sample_value。



    参数：

    key: list[str]

    用于从元戎数据系统中读取源数据的 key。



    返回值：

    sample_value: list[Any]

    读取的数据值。

    """



    def __write_data(

        self,

        key: list[str],

        sample_value: list[Any],

    ) -> bool:

    """

    功能描述：

    将 (key, sample_value) 写入元戎数据系统，并返回写入状态。



    参数：

    key: list[str]

    样本数据的 location key。



    sample_value: list[Any]

    写入的样本数据值。



    返回值：

    写入是否成功。

    """



    def _cast_value(

        self,

        column_name: str,

        column_value: list[Any],

    ) -> list[Any]:

    """

    功能描述：

    根据 column_name 和 column_type 将 column_value 转换为目标 Python 数据类型。



    参数：

    column_name: str

    需要转换的单个列名。



    column_value: list[Any]

    需要转换的单个列的值。



    返回值：

    converted_column_value: list[Any]

    返回转换后的单个列的值。

    """
