from .ae_ode import UserCNFTransform,ItemCNFTransform



def model_factory(args):
    if args.model_code=='ode':
        UserCNF=UserCNFTransform
        ItemCNF=ItemCNFTransform
        return (UserCNF(args),ItemCNF(args))
    else:
        return False