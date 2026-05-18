import timm

def create_model(model_name, num_classes, pretrained=True, drop_rate=0.0, drop_path_rate=0.0):
    """Factory function supporting baseline and advanced architectures."""
    if model_name == 'swin_tiny':
        model_type = 'swin_tiny_patch4_window7_224'
        return timm.create_model(model_type, pretrained=pretrained, num_classes=num_classes)
        
    elif model_name == 'convnext_tiny':
        model_type = 'convnext_tiny'
        return timm.create_model(model_type, pretrained=pretrained, num_classes=num_classes)
        
    elif model_name == 'convnext_base_advanced':
        model_type = 'convnext_base.fb_in22k_ft_in1k_384'
        return timm.create_model(model_type, pretrained=pretrained, num_classes=num_classes,
                                 drop_rate=drop_rate, drop_path_rate=drop_path_rate)
    else:
        raise ValueError(f"Model {model_name} not supported.")