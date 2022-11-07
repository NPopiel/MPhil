def __pick_cmap(map_name, cmap_dict):
    """Takes value from dictionary and returns value

    Parameters
    ----------
    map_name: str
        The unit chosen
    cmap_dict: dict
        The dictionary with the units and the values

    Returns
    -------
    cmap: list
        List of hexadecimal colour codes for the given map name
    """

    root_name, *r = map_name.split('_')

    if root_name in cmap_dict.keys():

        return cmap_dict[root_name]
    else:

        cmap_list = ["'{}',".format(x) for x in cmap_dict.keys()]
        cmap_list[-1] = "or {}.".format(cmap_list[-1][:-1])
        raise ValueError("Colormap must be in this list. The order can be reversed by appending '_r' to the name {}".format(" ".join(cmap_list)))


def select_discrete_cmap(map_name='charzard'):
    """
    Select a discrete Colormap to be used for plotting as a pokemon name.
    Pallettes generated using pokepalletes.com

    Parameters
    ----------
    map_name : str, optional
        The name of the pokemon with the corresponding colormap.

    Returns
    -------
    The list of hex-code strings of the colors in the colormap  : list
    """

    bulbasaur = ['#62d5b4', '#83eec5', '#184a4a', '#73ac31',
                 '#317373', '#bdff73', '#a4d541', '#ac0031',
                 '#526229', '#cdcdcd', '#ff6a62']

    venasaur = ['#105241', '#ff7b73', '#107b6a', '#5a9c39',
                '#5ad5c5', '#de4141', '#83de7b', '#833100',
                '#ffbdbd', '#bd6a31', '#ffee52', '#debd29']

    charzard = ['#cd5241', '#084152', '#833118', '#eede7b',
                '#207394', '#eeb45a', '#e64110', '#ffd510',
                '#f6a410', '#cdcdcd', '#626262']

    squirtle = ['#8bc5cd', '#297383', '#cd7b29', '#e6ac5a',
                '#ffe69c', '#ffd56a', '#832900', '#b4e6ee',
                '#d5cdcd', '#bd6a00', '#622900', '#d59452']

    wartortle = ['#627bc5', '#d5eef6', '#bdc5e6', '#dec58b',
                 '#29416a', '#ac8b62', '#acc5ff', '#6a4a18',
                 '#946262', '#31414a', '#8b5a20', '#d59452']

    blastoise = ['#083962', '#2062ac', '#d5ac4a', '#5a3918',
                 '#cdcdd5', '#4a4a4a', '#f6d59c', '#94ace6',
                 '#949494', '#8b6241', '#e6c573', '#d59452']

    caterpie = ['#e6cd94', '#6ad531', '#f6ee8b', '#185a41',
                '#52624a', '#207b4a', '#7b394a', '#ff4141',
                '#7b834a', '#acf641', '#4ac529', '#ffb439']

    butterfree = ['#313152', '#5a4a73', '#bdbde6', '#8383bd',
                  '#7362ac', '#6abdcd', '#527bc5', '#de3131',
                  '#623139', '#ff9cb4', '#9c9cb4', '#e66283']

    beedrill = ['#bdacc5', '#8b7b94', '#e69420', '#eee6ff',
                '#525a7b', '#ffcd4a', '#decdf6', '#fff6a4',
                '#834a00', '#b46210', '#9c0008', '#d51831',
                '#ff946a']

    pidgey = ['#e6bd62', '#412918', '#cd835a', '#ffee9c',
              '#734a31', '#bd2920', '#ee6241', '#fff6bd',
              '#838383', '#ffac73', '#bdbdbd', '#ff946a']

    pidegotto = ['#e6bd5a', '#835231', '#ac7b5a', '#c53129',
                 '#412920', '#7b3120', '#ee6241', '#de73cd',
                 '#ffac73', '#7b7b83', '#f69cee']

    rattata = ['#8b4a8b', '#d59cd5', '#4a2941', '#eedeb4',
               '#a47308', '#cdac62', '#624a08', '#e6cd73',
               '#e65a73', '#cdcdcd', '#5a5a5a', '#a41839']

    spearow = ['#9c5218', '#ac414a', '#8b7b62', '#c56a20',
               '#eeac52', '#6a4118', '#4a4139', '#ffd5cd',
               '#ffa48b', '#7b2929', '#c5b49c', '#e6d5b4']

    ekans = ['#a44a8b', '#eea4d5', '#5a104a', '#7b316a',
             '#e6ac5a', '#ffd56a', '#835210', '#b47b31',
             '#9c1000', '#f6734a', '#c54118', '#ffe69c']

    pikachu = ['#f6bd20', '#9c5200', '#de9400', '#623108',
               '#41414a', '#292929', '#fff6a4', '#c52018',
               '#e65a41', '#737383', '#ffe69c']

    nidoran = ['#62416a', '#006241', '#eebdee', '#9c4aac',
               '#00a473', '#18cd9c', '#cdcdcd', '#ff6a52',
               '#de4129', '#b41800']

    clefairy = ['#ffacac', '#9c5252', '#e67b7b', '#5a3120',
                '#734a39', '#949494', '#9c8373', '#b43929',
                '#ee4139', '#ac1000', '#e64131']

    jigglypuff = ['#e67383', '#a41020', '#ffcdc5', '#104a8b',
                  '#1873c5', '#6a5262', '#e6e6e6', '#cdeeff',
                  '#10b4ee', '#ff945a']

    zubat = ['#627bb4', '#4a417b', '#b4529c', '#bdcdff',
             '#73205a', '#d573bd', '#626262', '#d5d5d5',
             '#d5d5d5', '#9c6a20']

    gloom = ['#7b83a4', '#ac5239', '#833918', '#ff7300',
             '#c57329', '#8b3918', '#10314a', '#ffbd41',
             '#5a2900', '#cd734a', '#f6eebd', '#b45ac5']

    diglett = ['#c57341', '#a45a5a', '#837b4a', '#5a5220',
               '#e6e6b4', '#de9c5a', '#730018', '#5a3118',
               '#d5394a', '#ff6a5a', '#ffac94']


    pokedex =  {'bulbasaur':bulbasaur,
                'venasaur':venasaur,
                'charzard':charzard,
                'squirtle':squirtle,
                'wartortle':wartortle,
                'blastoise':blastoise,
                'caterpie':caterpie,
                'butterfree':butterfree,
                'beedrill':beedrill,
                'pidgey':pidgey,
                'pideotto':pidegotto,
                'rattata':rattata,
                'spearow':spearow,
                'ekans':ekans,
                'pikachu':pikachu,
                'nidoran':nidoran,
                'clefairy':clefairy,
                'jugglypuff':jigglypuff,
                'zubat':zubat,
                'gloom':gloom,
                'diglett':diglett}

    return __pick_cmap(map_name, pokedex)

