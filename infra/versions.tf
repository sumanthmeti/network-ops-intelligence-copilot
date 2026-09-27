terraform {
  required_version = ">= 1.6"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.0" # any 6.x; a 7.0 release could break things, so it needs a deliberate bump
    }
  }

  # State is local (infra/terraform.tfstate, gitignored). For a team you would use
  # an S3 backend with state locking so two people can't apply at once; for a
  # one-person portfolio project local state avoids a chicken-and-egg bootstrap.
}

provider "aws" {
  region = var.region

  # Stamped on every resource: lets Cost Explorer show exactly what this project costs,
  # and makes stray resources easy to find in the console.
  default_tags {
    tags = {
      Project   = var.project
      ManagedBy = "terraform"
    }
  }
}

# Who is running Terraform. Used to build a globally unique bucket name and to
# scope IAM policies to this account only.
data "aws_caller_identity" "current" {}
