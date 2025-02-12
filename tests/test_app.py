def test_app():
    from aiidalab_qe.app import App
    from aiida import load_profile, orm

    load_profile()

    pw_code = orm.load_code('pw-7.2@localhost')
    hp_code = orm.load_code('hp-7.2@localhost')

    app = App(qe_auto_setup=True)

    step1 = app.structure_step
    structure = step1.manager.children[0].children[3]
    # select LiCoO2
    structure.children[0].value = structure.children[0].options[10][1]
    step1.confirm()
    # step 2
    configure_step = app.configure_step
    parameters = {
        'workchain': {
            'relax_type': 'none',
            'electronic_type': 'insulator',
            'properties': ['hp'],
            'protocol': 'fast',
        }
    }
    configure_step.set_configuration_parameters(parameters)
    #
    configure_step.settings['hp'].calculation_type.value = 'DFT+U+V'
    configure_step.settings['hp'].qpoints_distance.value = 3.0
    configure_step.settings['hp'].hubbard_u_map['Co'][0].value = True
    configure_step.settings['hp'].hubbard_u_map['Co'][2].value = '3d'
    configure_step.settings['hp'].hubbard_u_map['Co'][3].value = 3.0
    configure_step.settings['hp'].hubbard_v_map[('Co', 'O')][0].value = True
    configure_step.settings['hp'].hubbard_v_map[('Co', 'O')][2].value = '3d'
    configure_step.settings['hp'].hubbard_v_map[('Co', 'O')][4].value = '2p'
    configure_step.confirm()
    #
    app.submit_step.pw_code.code_selection.refresh()
    app.submit_step.pw_code.code_selection.value = pw_code.uuid
    app.submit_step.pw_code.num_cpus.value = 4
    app.submit_step.codes['hp'].code_selection.value = hp_code.uuid
    app.submit_step.codes['hp'].num_cpus.value = 4
    # builder = app.submit_step._create_builder()
    # print("builder", builder)
    app.submit_step.submit()


def test_app_hubbard_u():
    from aiidalab_qe.app import App
    from aiida import load_profile, orm

    load_profile()

    pw_code = orm.load_code('pw-7.2@localhost')
    hp_code = orm.load_code('hp-7.2@localhost')

    app = App(qe_auto_setup=True)

    step1 = app.structure_step
    structure = step1.manager.children[0].children[3]
    # select LiCoO2
    structure.children[0].value = structure.children[0].options[10][1]
    step1.confirm()
    # step 2
    configure_step = app.configure_step
    parameters = {
        'workchain': {
            'relax_type': 'none',
            'electronic_type': 'insulator',
            'properties': ['hp'],
            'protocol': 'fast',
        }
    }
    configure_step.set_configuration_parameters(parameters)
    #
    configure_step.settings['hp'].hubbard_u_map['Co'][0].value = True
    configure_step.settings['hp'].hubbard_u_map['Co'][2].value = '3d'
    configure_step.settings['hp'].hubbard_u_map['Co'][3].value = 3.0
    configure_step.confirm()
    #
    app.submit_step.pw_code.code_selection.refresh()
    app.submit_step.pw_code.code_selection.value = pw_code.uuid
    app.submit_step.pw_code.num_cpus.value = 4
    app.submit_step.codes['hp'].code_selection.value = hp_code.uuid
    app.submit_step.codes['hp'].num_cpus.value = 4
    app.submit_step.submit()


def test_app_scf():
    from aiidalab_qe.app import App
    from aiida import load_profile, orm

    load_profile()

    pw_code = orm.load_code('pw-7.2@localhost')
    hp_code = orm.load_code('hp-7.2@localhost')

    app = App(qe_auto_setup=True)

    step1 = app.structure_step
    structure = step1.manager.children[0].children[3]
    # select LiCoO2
    structure.children[0].value = structure.children[0].options[10][1]
    step1.confirm()
    # step 2
    configure_step = app.configure_step
    parameters = {
        'workchain': {
            'relax_type': 'none',
            'electronic_type': 'insulator',
            'properties': ['hp'],
            'protocol': 'fast',
        }
    }
    configure_step.set_configuration_parameters(parameters)
    #
    configure_step.settings['hp'].method.value = 'self-consistent'
    configure_step.settings['hp'].calculation_type.value = 'DFT+U+V'
    configure_step.settings['hp'].hubbard_u_map['Co'][0].value = True
    configure_step.settings['hp'].hubbard_u_map['Co'][2].value = '3d'
    configure_step.settings['hp'].hubbard_u_map['Co'][3].value = 3.0
    configure_step.settings['hp'].hubbard_v_map[('Co', 'O')][0].value = True
    configure_step.settings['hp'].hubbard_v_map[('Co', 'O')][2].value = '3d'
    configure_step.settings['hp'].hubbard_v_map[('Co', 'O')][4].value = '2p'
    configure_step.confirm()
    #
    app.submit_step.pw_code.code_selection.refresh()
    app.submit_step.pw_code.code_selection.value = pw_code.uuid
    app.submit_step.pw_code.num_cpus.value = 4
    app.submit_step.codes['hp'].code_selection.value = hp_code.uuid
    app.submit_step.codes['hp'].num_cpus.value = 4
    app.submit_step.submit()
