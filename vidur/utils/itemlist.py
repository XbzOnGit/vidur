list_wrapper_id = 0
class ItemListWrapper:
    def __init__(self, item):
        global list_wrapper_id
        self._id = list_wrapper_id
        list_wrapper_id += 1
        self._item = item
        self._prev = None
        self._next = None
    @property
    def item(self):
        return self._item
    @property
    def prev(self):
        return self._prev
    @property
    def next(self):
        return self._next
    def set_prev(self, prev):
        self._prev = prev
    def set_next(self, next):
        self._next = next


list_id = 0
class ItemList:
    def __init__(self):
        global list_id
        self._id = list_id
        list_id += 1
        self._head = None
        self._tail = None
        self._chunk_cnt = 0
    def check_in_list(self, item: ItemListWrapper):
        if item.prev is None:
            return item is self._head
        else:
            return True
    def push_back(self, item: ItemListWrapper):
        if self._head is None:
            assert self._tail is None
            self._head = self._tail = item
            item.set_prev(None)
            item.set_next(None)
        else:
            assert self._tail is not None
            self._tail.set_next(item)
            item.set_prev(self._tail)
            self._tail = item
        self._chunk_cnt += 1
        # print(f"{self._id}: list size {self._chunk_cnt - 1} --> {self._chunk_cnt}")
    def remove(self, wrapped_item: ItemListWrapper):
        # Assuming inside then remove.
        if wrapped_item.prev is None:
            assert wrapped_item is self._head
            self._head = wrapped_item.next
        else:
            wrapped_item.prev.set_next(wrapped_item.next)
        if wrapped_item.next is None:
            assert wrapped_item is self._tail
            self._tail = wrapped_item.prev
        else:
            wrapped_item.next.set_prev(wrapped_item.prev)
        wrapped_item.set_next(None)
        wrapped_item.set_prev(None)
        self._chunk_cnt -= 1
        # print(f"{self._id}: list size {self._chunk_cnt + 1} --> {self._chunk_cnt}")
    def pop_front(self):
        if self._head is None:
            return None
        assert self._head is not None
        assert self._tail is not None
        wrapped_item = self._head
        assert wrapped_item is not None
        self.remove(wrapped_item)
        return wrapped_item.item