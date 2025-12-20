import React from 'react';
import Navbar from '@theme-original/Navbar';
import AuthNavbar from '@site/src/components/auth/AuthNavbar';

const NavbarWrapper = (props) => {
  return (
    <>
      <Navbar {...props} />
      <div className="navbar__item navbar__right">
        <AuthNavbar />
      </div>
    </>
  );
};

export default NavbarWrapper;