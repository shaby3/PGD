_base_ = "base/1x_setting.py"

alpha       = 0.13
# beta        = alpha * 0.5
beta        = alpha * 0.5

distiller = dict(
    type='PredictionGuidedDistiller',
    teacher_pretrained = 'work_dirs/atss_r101_3x_ms/epoch_36.pth',
    init_student = True,
    distill_cfg=[dict(student_module='bbox_head.cls_identities.0',
                      teacher_module='bbox_head.cls_identities.0',
                      output_hook=True,
                      methods=[dict(type='PGDClsLoss',
                                    name='loss_kd_cls_0',
                                    loss_weight=1.0,
                                    alpha=alpha,
                                    beta=beta,
                                    )
                               ]
                      ),
                 dict(student_module='bbox_head.cls_identities.1',
                      teacher_module='bbox_head.cls_identities.1',
                      output_hook=True,
                      methods=[dict(type='PGDClsLoss',
                                    name='loss_kd_cls_1',
                                    loss_weight=1.0,
                                    alpha=alpha,
                                    beta=beta,
                                    )
                               ]
                      ),
                 dict(student_module='bbox_head.cls_identities.2',
                      teacher_module='bbox_head.cls_identities.2',
                      output_hook=True,
                      methods=[dict(type='PGDClsLoss',
                                    name='loss_kd_cls_2',
                                    loss_weight=1.0,
                                    alpha=alpha,
                                    beta=beta,
                                    )
                               ]
                      ),
                 dict(student_module='bbox_head.cls_identities.3',
                      teacher_module='bbox_head.cls_identities.3',
                      output_hook=True,
                      methods=[dict(type='PGDClsLoss',
                                    name='loss_kd_cls_3',
                                    loss_weight=1.0,
                                    alpha=alpha,
                                    beta=beta,
                                    )
                               ]
                      ),
                 dict(student_module='bbox_head.cls_identities.4',
                      teacher_module='bbox_head.cls_identities.4',
                      output_hook=True,
                      methods=[dict(type='PGDClsLoss',
                                    name='loss_kd_cls_4',
                                    loss_weight=1.0,
                                    alpha=alpha,
                                    beta=beta,
                                    )
                               ]
                      ),
                 dict(student_module='bbox_head.reg_identities.0',
                      teacher_module='bbox_head.reg_identities.0',
                      output_hook=True,
                      methods=[dict(type='PGDRegLoss',
                                    name='loss_kd_reg_0',
                                    loss_weight=1.0,
                                    alpha=alpha,
                                    beta=beta,
                                    )
                               ]
                      ),
                 dict(student_module='bbox_head.reg_identities.1',
                      teacher_module='bbox_head.reg_identities.1',
                      output_hook=True,
                      methods=[dict(type='PGDRegLoss',
                                    name='loss_kd_reg_1',
                                    loss_weight=1.0,
                                    alpha=alpha,
                                    beta=beta,
                                    )
                               ]
                      ),
                 dict(student_module='bbox_head.reg_identities.2',
                      teacher_module='bbox_head.reg_identities.2',
                      output_hook=True,
                      methods=[dict(type='PGDRegLoss',
                                    name='loss_kd_reg_2',
                                    loss_weight=1.0,
                                    alpha=alpha,
                                    beta=beta,
                                    )
                               ]
                      ),
                 dict(student_module='bbox_head.reg_identities.3',
                      teacher_module='bbox_head.reg_identities.3',
                      output_hook=True,
                      methods=[dict(type='PGDRegLoss',
                                    name='loss_kd_reg_3',
                                    loss_weight=1.0,
                                    alpha=alpha,
                                    beta=beta,
                                    )
                               ]
                      ),
                 dict(student_module='bbox_head.reg_identities.4',
                      teacher_module='bbox_head.reg_identities.4',
                      output_hook=True,
                      methods=[dict(type='PGDRegLoss',
                                    name='loss_kd_reg_4',
                                    loss_weight=1.0,
                                    alpha=alpha,
                                    beta=beta,
                                    )
                               ]
                      ),
                ]
    )

student_cfg = 'work_configs/detectors/atss_r50_distill_head.py'
teacher_cfg = 'work_configs/detectors/atss_r101_3x_ms.py'
