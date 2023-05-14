'''
For simplifying plot updating.
'''

import finplot


class Live:
    def __init__(self):
        self.colors = {}
        self.item = None
        self.item_create_func_name = ''


    def item_create_call(self, name):
        val = getattr(finplot, name)
        if not callable(val):
            return val
        def wrap_call(*args, **kwargs):
            if 'gfx' in kwargs: # only used in subsequent update calls
                del kwargs['gfx']
            item = val(*args, **kwargs)
            if isinstance(item, finplot.pg.GraphicsObject): # some kind of plot?
                setattr(self, 'item', item)
                setattr(self, 'item_create_func_name', val.__name__)
                # update to the "root" colors dict, if set
                if self.colors:
                    item.colors.update(self.colors)
                # from hereon, use the item.colors instead, if present
                if hasattr(item, 'colors'):
                    setattr(self, 'colors', item.colors)
                return self
            return item
        return wrap_call


    def item_update_call(self, name):
        if name == self.item_create_func_name:
            def wrap_call(*args, **kwargs):
                item = object.__getattribute__(self, 'item')
                ka = {'gfx':kwargs.get('gfx', True)} # only gfx parameter used
                assert len(args) == 1, 'only one unnamed argument allowed for live plots'
                item.update_data(*args, **ka)
            return wrap_call
        return getattr(self.item, name)


    def __getattribute__(self, name):
        try:
            return object.__getattribute__(self, name)
        except:
            pass
        # check if we're creating the plot, or updating data on an already created one
        if object.__getattribute__(self, 'item') is None:
            return self.item_create_call(name)
        return self.item_update_call(name)
